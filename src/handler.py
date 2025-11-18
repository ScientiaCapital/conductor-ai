import os
import runpod
from utils import JobInput, create_error_response
from engine import vLLMEngine, OpenAIvLLMEngine
from metering import get_meter, RequestTimer
from tracing import get_logger, setup_structured_logging, ErrorCodes
from health import get_health_checker, handle_health_request
from auth import get_authenticator
from metrics import get_metrics_collector, handle_metrics_request
from cache import get_cache, handle_cache_stats_request
from priority import get_priority_queue, handle_queue_stats_request, PriorityLevel
from dlq import handle_dlq_stats_request, handle_dlq_list_request, handle_dlq_retry_request
from validation import validate_config

# Initialize structured logging
setup_structured_logging()
logger = get_logger()

# Validate configuration before initializing engines
logger.info("Validating configuration...")
if not validate_config():
    logger.error("Configuration validation failed - some features may not work correctly")

# Initialize engines
vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

# Initialize enterprise features
health_checker = get_health_checker()
health_checker.set_engine(vllm_engine)

metrics_collector = get_metrics_collector()
metrics_collector.set_model_info(vllm_engine.engine_args.model)

authenticator = get_authenticator()
cache = get_cache()
priority_queue = get_priority_queue()

logger.info("Worker initialized",
    model=vllm_engine.engine_args.model,
    auth_enabled=authenticator.enabled,
    cache_enabled=cache.enabled,
    metrics_enabled=metrics_collector.enabled,
    priority_queue_enabled=priority_queue.enabled
)

async def handler(job):
    job_input = JobInput(job["input"])

    # Handle special routes first (no auth/metering for these)
    if job_input.openai_route == "/health":
        yield handle_health_request("/health")
        return
    elif job_input.openai_route == "/ready":
        yield handle_health_request("/ready")
        return
    elif job_input.openai_route == "/model":
        yield handle_health_request("/model")
        return
    elif job_input.openai_route == "/metrics":
        yield {"metrics": handle_metrics_request()}
        return
    elif job_input.openai_route == "/cache/stats":
        yield handle_cache_stats_request()
        return
    elif job_input.openai_route == "/queue/stats":
        yield handle_queue_stats_request()
        return
    elif job_input.openai_route == "/dlq/stats":
        yield handle_dlq_stats_request()
        return
    elif job_input.openai_route == "/dlq/list":
        limit = 100
        if job_input.openai_input:
            limit = job_input.openai_input.get("limit", 100)
        yield handle_dlq_list_request(limit)
        return
    elif job_input.openai_route == "/dlq/retry":
        message_id = None
        if job_input.openai_input:
            message_id = job_input.openai_input.get("message_id")
        result = await handle_dlq_retry_request(message_id)
        yield result
        return

    # Authentication check
    api_key = None
    if job_input.openai_input:
        # Check for API key in headers or input
        api_key = job_input.openai_input.get("api_key") or job_input.openai_input.get("authorization")

    auth_result = authenticator.authenticate(api_key)
    if not auth_result.authenticated:
        yield create_error_response(
            auth_result.error_message or "Authentication failed",
            err_type=auth_result.error_code or "AuthenticationError"
        ).model_dump()
        return

    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine

    # Initialize metering and tracing
    meter = get_meter()
    timer = RequestTimer().start()

    # Set correlation ID for request tracing
    logger.set_correlation_id(job_input.request_id)

    # Record metrics start
    metrics_collector.record_request_start()

    # Determine model name early for logging
    model_name = ""
    messages = None
    sampling_params = {}

    if job_input.openai_route and job_input.openai_input:
        model_name = job_input.openai_input.get("model", "")
        messages = job_input.openai_input.get("messages") or job_input.openai_input.get("prompt")
        sampling_params = {
            "temperature": job_input.openai_input.get("temperature", 1.0),
            "top_p": job_input.openai_input.get("top_p", 1.0),
            "max_tokens": job_input.openai_input.get("max_tokens"),
            "seed": job_input.openai_input.get("seed"),
        }
    else:
        messages = job_input.llm_input

    if not model_name:
        model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE", "") or vllm_engine.engine_args.model

    # Determine priority based on user tier
    user_tier = auth_result.key_info.tier if auth_result.key_info else "free"
    priority = PriorityLevel.from_tier(user_tier)

    # Log request start
    logger.request_start(
        request_id=job_input.request_id,
        route=job_input.openai_route or "native",
        model=model_name,
        stream=job_input.stream,
        user_id=auth_result.key_info.user_id if auth_result.key_info else None,
        tier=user_tier,
        priority=priority.value
    )

    # Check cache first
    cached_response, cache_hit = cache.get(
        model=model_name,
        messages=messages,
        sampling_params=sampling_params,
        stream=job_input.stream
    )

    if cache_hit:
        logger.info("Cache hit", request_id=job_input.request_id)
        timer.stop()

        # Return cached response
        yield cached_response

        # Record metrics for cache hit
        metrics_collector.record_request_complete(
            latency_ms=timer.latency_ms,
            time_to_first_token_ms=0,
            input_tokens=0,
            output_tokens=0,
            success=True
        )
        return

    # Track usage metrics
    total_input_tokens = 0
    total_output_tokens = 0
    is_first_batch = True
    success = True
    error_message = None
    error_code = None
    all_batches = []  # Collect for caching

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

            # Collect batches for caching (non-streaming only)
            if not job_input.stream:
                all_batches.append(batch)

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
            error_code=error_code,
            user_id=auth_result.key_info.user_id if auth_result.key_info else None
        )

        # Record metrics
        metrics_collector.record_request_complete(
            latency_ms=timer.latency_ms,
            time_to_first_token_ms=timer.time_to_first_token_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            success=success,
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
            user_id=auth_result.key_info.user_id if auth_result.key_info else None,
            organization_id=auth_result.key_info.organization_id if auth_result.key_info else None,
            api_key_id=auth_result.key_info.key_id if auth_result.key_info else None,
        )

        await meter.record_usage(record)

        # Record auth usage for rate limiting
        if auth_result.key_info:
            authenticator.record_usage(
                auth_result.key_info.key_id,
                total_input_tokens + total_output_tokens
            )

        # Cache successful responses
        if success and all_batches and not job_input.stream:
            # Cache the last batch (which contains the full response)
            cache.put(
                model=model_name,
                messages=messages,
                sampling_params=sampling_params,
                response=all_batches[-1] if len(all_batches) == 1 else all_batches,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                stream=job_input.stream
            )

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
