import os
import time
import torch
import ray
import io
import logging
import base64
import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
import argparse

from xfuser import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserFluxPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserHunyuanDiTPipeline,
    xFuserArgs,
)
# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = 42
    cfg: Optional[float] = 7.5
    save_disk_path: Optional[str] = None
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    metadata: Optional[Dict[str, Any]] = None

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "num_inference_steps": 50,
                "seed": 42,
                "cfg": 7.5,
                "height": 1024,
                "width": 1024
            }
        }

app = FastAPI()
logger = logging.getLogger(__name__)

SERVER_LAUNCH_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
_PROFILE_LOG_FILENAME = f"request-exec-info-{SERVER_LAUNCH_TIME}.jsonl"
def _resolve_profile_dir(base_dir: Optional[str]) -> Path:
    """
    Normalize the requested directory so it works on both POSIX and Windows.
    """
    target = base_dir or os.environ.get("REQUEST_PROFILE_LOG_DIR") or os.getcwd()
    if os.name != "nt" and "\\" in target:
        target = target.replace("\\", "/")
    return Path(target).expanduser()


_profile_log_dir = _resolve_profile_dir(None)
PROFILE_LOG_PATH = _profile_log_dir / _PROFILE_LOG_FILENAME
_profile_write_lock = threading.Lock()


def set_profile_log_dir(base_dir: Optional[str]):
    """
    Update the profiling log directory (must be called before requests arrive).
    """
    global _profile_log_dir, PROFILE_LOG_PATH
    _profile_log_dir = _resolve_profile_dir(base_dir)
    PROFILE_LOG_PATH = _profile_log_dir / _PROFILE_LOG_FILENAME


def _to_utc_iso(ts: Optional[Union[float, datetime]]) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _write_profile_line(line: str):
    PROFILE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _profile_write_lock:
        with open(PROFILE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


async def _log_request_profile(record: Dict[str, Any]):
    line = json.dumps(record, ensure_ascii=False, default=str)
    await asyncio.to_thread(_write_profile_line, line)

@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args : xFuserArgs):

        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        
        model_name = self.engine_config.model_config.model.split("/")[-1]
        pipeline_map = {
            "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
            "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
            "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
            "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
            "FLUX.1-schnell": xFuserFluxPipeline,
            "FLUX.1-dev": xFuserFluxPipeline,
        }
        
        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")

        self.logger.info(f"Initializing model {model_name} from {xfuser_args.model}")

        self.pipe = PipelineClass.from_pretrained(
            pretrained_model_name_or_path=xfuser_args.model,
            engine_config=self.engine_config,
            torch_dtype=torch.float16,
        ).to("cuda")
        
        self.pipe.prepare_run(self.input_config)
        self.logger.info("Model initialization completed")

    def generate(self, request: GenerateRequest):
        try:
            execution_started_at = time.time()
            output = self.pipe(
                height=request.height,
                width=request.width,
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                output_type="pil",
                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                guidance_scale=request.cfg,
                max_sequence_length=self.input_config.max_sequence_length
            )
            execution_completed_at = time.time()
            elapsed_time = execution_completed_at - execution_started_at

            if self.pipe.is_dp_last_group():
                if request.save_disk_path:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"generated_image_{timestamp}.png"
                    file_path = os.path.join(request.save_disk_path, filename)
                    os.makedirs(request.save_disk_path, exist_ok=True)
                    output.images[0].save(file_path)
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True,
                        "execution_started_at": execution_started_at,
                        "execution_completed_at": execution_completed_at,
                        "result_file_name": filename,
                        "result_file_path": file_path,
                    }
                else:
                    # Convert to base64
                    buffered = io.BytesIO()
                    output.images[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return {
                        "message": "Image generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": img_str,
                        "save_to_disk": False,
                        "execution_started_at": execution_started_at,
                        "execution_completed_at": execution_completed_at,
                        "result_file_name": None,
                        "result_file_path": None,
                    }
            return None

        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()
        
        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(xfuser_args, rank=rank, world_size=world_size)
            for rank in range(num_workers)
        ]
        
    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None) 

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    arrival_time = time.time()
    result = None
    error_detail: Optional[str] = None
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(status_code=400, detail="Height and width must be positive")
        if request.num_inference_steps <= 0:
            raise HTTPException(status_code=400, detail="num_inference_steps must be positive")

        result = await engine.generate(request)
        return result
    except Exception as e:
        error_detail = str(e)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        completion_time = time.time()
        execution_started_at = None
        execution_completed_at = None
        result_file_name = None
        result_file_path = None
        if isinstance(result, dict):
            execution_started_at = result.get("execution_started_at")
            execution_completed_at = result.get("execution_completed_at")
            result_file_name = result.get("result_file_name")
            result_file_path = result.get("result_file_path")

        profile_record = {
            "server_launch_time": SERVER_LAUNCH_TIME,
            "arrival_time": arrival_time,
            "exec_start_time": execution_started_at,
            "completion_time": completion_time,
            "engine_exec_end_time": execution_completed_at,
            "request_metadata": request.metadata,
            "request_payload": request.dict(),
            "result_file_name": result_file_name,
            "result_file_path": result_file_path,
            "status": "success" if error_detail is None else "error",
            "error_detail": error_detail,
        }
        try:
            await _log_request_profile(profile_record)
        except Exception as log_error:
            logger.warning("Failed to write profiling data: %s", log_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--pipefusion_parallel_degree', type=int, default=1, help='Degree of pipeline fusion parallelism')
    parser.add_argument('--ulysses_parallel_degree', type=int, default=1, help='Degree of Ulysses parallelism')
    parser.add_argument('--data_parallel_degree', type=int, default=1, help='Degree of data parallel replicas')
    parser.add_argument('--save_disk_path', type=str, default='output', help='Path to save generated images')
    parser.add_argument('--use_cfg_parallel', action='store_true', help='Whether to use CFG parallel')
    args = parser.parse_args()

    set_profile_log_dir(args.save_disk_path)

    xfuser_args = xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=1,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=args.ulysses_parallel_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        data_parallel_degree=args.data_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        dit_parallel_size=0,
    )
    
    engine = Engine(
        world_size=args.world_size,
        xfuser_args=xfuser_args
    )
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
