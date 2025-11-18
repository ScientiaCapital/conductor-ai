import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine
from metering import get_meter, RequestTimer
from tracing import get_logger, setup_structured_logging, ErrorCodes
from health import get_health_checker, handle_health_request

# Initialize structured logging
setup_structured_logging()
logger = get_logger()

# Initialize engines
vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

# Register engine with health checker
health_checker = get_health_checker()
health_checker.set_engine(vllm_engine)
logger.info("Worker initialized", model=vllm_engine.engine_args.model)

async def handler(job):
    job_input = JobInput(job["input"])

    # Handle health check routes first (no metering/tracing for these)
    if job_input.openai_route in ["/health", "/ready", "/model"]:
        yield handle_health_request(job_input.openai_route)
        return

    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine

    # Initialize metering and tracing
    meter = get_meter()
    timer = RequestTimer().start()

    # Set correlation ID for request tracing
    logger.set_correlation_id(job_input.request_id)

    # Determine model name early for logging
    model_name = ""
    if job_input.openai_route and job_input.openai_input:
        model_name = job_input.openai_input.get("model", "")
    if not model_name:
        model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE", "") or vllm_engine.engine_args.model

    # Log request start
    logger.request_start(
        request_id=job_input.request_id,
        route=job_input.openai_route or "native",
        model=model_name,
        stream=job_input.stream
    )

    # Track usage metrics
    total_input_tokens = 0
    total_output_tokens = 0
    is_first_batch = True
    success = True
    error_message = None
    error_code = None

    try:
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            # Mark first token time
            if is_first_batch:
                timer.mark_first_token()
                is_first_batch = False

            # Extract token counts from batch
            if isinstance(batch, dict):
                if "error" in batch:
                    success = False
                    error_message = batch.get("error", {}).get("message", "Unknown error")
                    error_code = ErrorCodes.categorize_error(error_message)
                elif "usage" in batch:
                    usage = batch["usage"]
                    total_input_tokens = usage.get("input", 0)
                    total_output_tokens = usage.get("output", 0)

            yield batch

    except Exception as e:
        success = False
        error_message = str(e)
        error_code = ErrorCodes.categorize_error(error_message)
        logger.request_error(
            request_id=job_input.request_id,
            error_code=error_code,
            error_message=error_message
        )
        raise
    finally:
        # Record usage after request completes
        timer.stop()

        # Determine model type
        model_type = "completion"
        if job_input.openai_route:
            model_type = "chat" if "chat" in job_input.openai_route else "completion"

        # Log request completion
        logger.request_complete(
            request_id=job_input.request_id,
            latency_ms=timer.latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            success=success,
            time_to_first_token_ms=timer.time_to_first_token_ms,
            model=model_name,
            error_code=error_code
        )

        # Create and record usage for billing
        record = meter.create_record(
            request_id=job_input.request_id,
            model=model_name,
            model_type=model_type,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=timer.latency_ms,
            time_to_first_token_ms=timer.time_to_first_token_ms,
            stream=job_input.stream,
            success=success,
            error_code=error_code,
            error_message=error_message,
        )

        await meter.record_usage(record)

        # Record request for health metrics
        health_checker.record_request(success=success)

        # Clear correlation ID
        logger.clear_correlation_id()

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)