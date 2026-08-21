from types import SimpleNamespace

import pytest

from litellm.router_utils.search_api_router import SearchAPIRouter


@pytest.mark.asyncio
async def test_search_tool_litellm_params_are_forwarded_to_selected_provider():
    captured_kwargs: dict[str, object] = {}

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "search response"

    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "configured-key",
                    "api_base": "https://configured.example.com/mcp",
                    "tool_name": "configured___WebSearch",
                },
            }
        ]
    )

    response = await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=search_provider,
        query="LiteLLM",
    )

    assert response == "search response"
    assert captured_kwargs == {
        "search_provider": "agentcore",
        "api_key": "configured-key",
        "api_base": "https://configured.example.com/mcp",
        "tool_name": "configured___WebSearch",
        "query": "LiteLLM",
    }


@pytest.mark.asyncio
async def test_search_request_params_override_search_tool_litellm_params():
    captured_kwargs: dict[str, object] = {}

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "search response"

    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "configured-key",
                    "api_base": "https://configured.example.com/mcp",
                    "tool_name": "configured___WebSearch",
                },
            }
        ]
    )

    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=search_provider,
        query="LiteLLM",
        api_key="request-key",
        api_base="https://request.example.com/mcp",
        tool_name="request___WebSearch",
    )

    assert captured_kwargs["api_key"] == "request-key"
    assert captured_kwargs["api_base"] == "https://request.example.com/mcp"
    assert captured_kwargs["tool_name"] == "request___WebSearch"


@pytest.mark.asyncio
async def test_fallback_search_tool_receives_its_merged_provider_params():
    captured_kwargs: dict[str, object] = {}

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "fallback response"

    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "primary-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "api_key": "primary-key",
                    "tool_name": "primary___WebSearch",
                },
            },
            {
                "search_tool_name": "fallback-search",
                "litellm_params": {
                    "search_provider": "exa",
                    "api_key": "fallback-key",
                    "api_base": "https://fallback.example.com/search",
                    "max_results": 5,
                },
            },
        ]
    )

    response = await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="fallback-search",
        original_generic_function=search_provider,
        query="LiteLLM",
        search_provider="agentcore",
        max_results=8,
    )

    assert response == "fallback response"
    assert captured_kwargs == {
        "search_provider": "exa",
        "api_key": "fallback-key",
        "api_base": "https://fallback.example.com/search",
        "max_results": 8,
        "query": "LiteLLM",
    }, f"fallback provider received unexpected kwargs: {captured_kwargs!r}"


@pytest.mark.asyncio
async def test_unknown_search_tool_param_is_rejected_without_exposing_its_value(caplog: pytest.LogCaptureFixture):
    captured_kwargs: dict[str, object] = {}
    secret = "configured-secret-do-not-log"

    async def search_provider(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "search response"

    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "agentcore-search",
                "litellm_params": {
                    "search_provider": "agentcore",
                    "tool_name": "configured___WebSearch",
                    "unknown_provider_param": secret,
                },
            }
        ]
    )

    with pytest.raises(ValueError, match="unknown_provider_param") as exc_info:
        await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="agentcore-search",
            original_generic_function=search_provider,
            query="LiteLLM",
        )

    assert captured_kwargs == {}, f"unknown YAML key was silently forwarded in kwargs: {captured_kwargs!r}"
    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
