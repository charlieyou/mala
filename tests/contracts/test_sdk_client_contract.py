"""Contract tests for SDKClientProtocol and SDKClientFactoryProtocol implementations.

Ensures FakeSDKClient and FakeSDKClientFactory implement all methods of their
respective protocols and exhibit correct behavioral parity.
"""

import pytest

from src.core.protocols.sdk import SDKClientFactoryProtocol, SDKClientProtocol
from tests.contracts import get_protocol_members
from tests.fakes.sdk_client import FakeSDKClient, FakeSDKClientFactory


@pytest.mark.unit
def test_fake_sdk_client_implements_all_protocol_methods() -> None:
    """FakeSDKClient must implement all public methods of SDKClientProtocol."""
    protocol_methods = get_protocol_members(SDKClientProtocol)
    fake_methods = {name for name in dir(FakeSDKClient) if not name.startswith("_")}

    # Filter out dunder methods from protocol (like __aenter__, __aexit__)
    # that appear with underscores in dir() but are in protocol_methods without
    public_protocol = {m for m in protocol_methods if not m.startswith("_")}
    dunder_protocol = protocol_methods - public_protocol

    # Check public methods
    missing_public = public_protocol - fake_methods
    assert not missing_public, (
        f"FakeSDKClient missing protocol methods: {sorted(missing_public)}"
    )

    # Check dunder methods exist
    for dunder in dunder_protocol:
        assert hasattr(FakeSDKClient, dunder), (
            f"FakeSDKClient missing protocol method: {dunder}"
        )


@pytest.mark.unit
def test_fake_sdk_client_protocol_compliance() -> None:
    """FakeSDKClient passes runtime isinstance check for SDKClientProtocol."""
    client = FakeSDKClient()
    assert isinstance(client, SDKClientProtocol)


@pytest.mark.unit
def test_fake_sdk_client_factory_implements_all_protocol_methods() -> None:
    """FakeSDKClientFactory must implement all public methods of SDKClientFactoryProtocol."""
    protocol_methods = get_protocol_members(SDKClientFactoryProtocol)
    fake_methods = {
        name for name in dir(FakeSDKClientFactory) if not name.startswith("_")
    }

    missing = protocol_methods - fake_methods
    assert not missing, (
        f"FakeSDKClientFactory missing protocol methods: {sorted(missing)}"
    )


@pytest.mark.unit
def test_fake_sdk_client_factory_protocol_compliance() -> None:
    """FakeSDKClientFactory passes runtime isinstance check for SDKClientFactoryProtocol."""
    factory = FakeSDKClientFactory()
    assert isinstance(factory, SDKClientFactoryProtocol)


class TestFakeSDKClientBehavior:
    """Behavioral tests for FakeSDKClient."""

    @pytest.mark.unit
    async def test_context_manager_protocol(self) -> None:
        """FakeSDKClient works as async context manager."""
        client = FakeSDKClient()
        async with client as ctx:
            assert ctx is client

    @pytest.mark.unit
    async def test_query_records_calls(self) -> None:
        """query() records prompt and session_id."""
        client = FakeSDKClient()
        await client.query("Hello, world!", session_id="session-1")
        assert len(client.queries) == 1
        assert client.queries[0] == ("Hello, world!", "session-1")

    @pytest.mark.unit
    async def test_query_raises_configured_error(self) -> None:
        """query() raises configured query_error."""
        error = RuntimeError("API error")
        client = FakeSDKClient(query_error=error)
        with pytest.raises(RuntimeError, match="API error"):
            await client.query("test")

    @pytest.mark.unit
    async def test_receive_response_yields_messages(self) -> None:
        """receive_response() yields configured messages without SDK dependency."""
        msg1 = {"type": "text", "content": "Hello"}
        msg2 = {"type": "text", "content": "World"}
        # Set result_message=None to avoid SDK import in unit test
        client = FakeSDKClient(messages=[msg1, msg2], result_message=None)

        messages = []
        async for msg in client.receive_response():
            messages.append(msg)

        # Should have only configured messages (no ResultMessage)
        assert len(messages) == 2
        assert messages[0] == msg1
        assert messages[1] == msg2

    @pytest.mark.integration
    async def test_receive_response_yields_default_result_message(self) -> None:
        """receive_response() yields default ResultMessage when not configured.

        Note: Marked as integration test because it requires claude_agent_sdk import.
        """
        client = FakeSDKClient(messages=[])

        messages = []
        async for msg in client.receive_response():
            messages.append(msg)

        # Should have one default ResultMessage
        assert len(messages) == 1
        assert hasattr(messages[0], "subtype")
        assert messages[0].subtype == "result"

    @pytest.mark.unit
    async def test_receive_response_no_result_when_explicitly_none(self) -> None:
        """receive_response() yields no ResultMessage when result_message=None."""
        client = FakeSDKClient(messages=[{"type": "text"}], result_message=None)

        messages = []
        async for msg in client.receive_response():
            messages.append(msg)

        # Should only have the configured message, no ResultMessage
        assert len(messages) == 1
        assert messages[0] == {"type": "text"}

    @pytest.mark.unit
    async def test_disconnect_sets_flag(self) -> None:
        """disconnect() sets disconnect_called flag."""
        client = FakeSDKClient()
        assert client.disconnect_called is False
        await client.disconnect()
        assert client.disconnect_called is True

    @pytest.mark.unit
    def test_backward_compatibility_aliases(self) -> None:
        """Backward compatibility aliases work correctly."""
        client = FakeSDKClient()
        # _query_calls should alias queries - verify mutation affects both
        client.queries.append(("test", None))
        assert ("test", None) in client._query_calls
        # _disconnect_called should alias disconnect_called
        assert client._disconnect_called == client.disconnect_called


class TestFakeSDKClientFactoryBehavior:
    """Behavioral tests for FakeSDKClientFactory."""

    @pytest.mark.unit
    def test_create_returns_configured_client(self) -> None:
        """create() returns pre-configured single client."""
        client = FakeSDKClient()
        factory = FakeSDKClientFactory(client=client)
        result = factory.create({})
        assert result is client

    @pytest.mark.unit
    def test_create_returns_queued_clients(self) -> None:
        """create() returns clients from queue in order."""
        factory = FakeSDKClientFactory()
        client1 = factory.configure_next_client(messages=[{"id": 1}])
        client2 = factory.configure_next_client(messages=[{"id": 2}])

        result1 = factory.create({})
        result2 = factory.create({})

        assert result1 is client1
        assert result2 is client2

    @pytest.mark.unit
    def test_create_returns_new_client_when_queue_empty(self) -> None:
        """create() returns new FakeSDKClient when queue is empty."""
        factory = FakeSDKClientFactory()
        result = factory.create({})
        assert isinstance(result, FakeSDKClient)
        assert result in factory.clients

    @pytest.mark.unit
    def test_create_records_calls(self) -> None:
        """create() records all options passed."""
        factory = FakeSDKClientFactory()
        options = {"cwd": "/tmp", "model": "opus"}
        factory.create(options)
        assert options in factory.create_calls

    @pytest.mark.unit
    def test_create_options_returns_dict(self) -> None:
        """create_options() returns options dict and records call."""
        factory = FakeSDKClientFactory()
        opts = factory.create_options(
            cwd="/home/user/project",
            model="sonnet",
            system_prompt={"text": "You are a helpful assistant"},
        )
        assert opts["cwd"] == "/home/user/project"
        assert opts["model"] == "sonnet"
        assert "settings" in opts
        assert len(factory.created_options) == 1

    @pytest.mark.unit
    def test_with_resume_creates_copy(self) -> None:
        """with_resume() creates copy of options with resume field."""
        factory = FakeSDKClientFactory()
        original = {"cwd": "/tmp", "model": "opus"}
        result = factory.with_resume(original, resume="session-123")

        assert result["resume"] == "session-123"
        assert result["cwd"] == "/tmp"
        # Original should be unchanged
        assert "resume" not in original or original.get("resume") != "session-123"

    @pytest.mark.unit
    def test_with_resume_records_calls(self) -> None:
        """with_resume() records all calls."""
        factory = FakeSDKClientFactory()
        opts = {"cwd": "/tmp"}
        factory.with_resume(opts, resume="session-1")
        factory.with_resume(opts, resume=None)

        assert len(factory.with_resume_calls) == 2
        assert factory.with_resume_calls[0] == (opts, "session-1")
        assert factory.with_resume_calls[1] == (opts, None)

    @pytest.mark.unit
    def test_create_hook_matcher_returns_tuple(self) -> None:
        """create_hook_matcher() returns tuple and records call."""
        factory = FakeSDKClientFactory()
        matcher = {"pattern": "test"}
        hooks = [lambda x: x]
        result = factory.create_hook_matcher(matcher, hooks)

        assert result == ("matcher", matcher, hooks)
        assert len(factory.created_matchers) == 1
        assert factory.created_matchers[0] == result

    @pytest.mark.unit
    def test_configure_next_client_with_error(self) -> None:
        """configure_next_client() configures client with query_error."""
        factory = FakeSDKClientFactory()
        error = ValueError("test error")
        client = factory.configure_next_client(query_error=error)
        assert client.query_error is error
