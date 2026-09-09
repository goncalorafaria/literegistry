import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import literegistry_tool_client
from literegistry_tool_client import SearchClient, WebAssetStore


def test_import_without_training_or_server_packages():
    code = """
import sys
class BlockTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'primebeaker', 'literegistry', 'verifiers', 'torch'}:
            raise AssertionError(f'Unexpected dependency: {fullname}')
sys.meta_path.insert(0, BlockTrainingImports())
import literegistry_tool_client as client
for name in client.__all__:
    assert getattr(client, name) is not None
"""
    subprocess.run(
        [sys.executable, '-c', code], check=True,
        env={**os.environ, 'PYTHONPATH': str(Path(literegistry_tool_client.__file__).parent.parent)},
    )


def test_search_sends_gateway_payload_and_preserves_response():
    class Response:
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def text(self):
            return json.dumps({'results': [{'url': 'https://example.org'}]})

    class Session:
        def post(self, url, **kwargs):
            self.url, self.kwargs = url, kwargs
            return Response()

    client = SearchClient('http://gateway:1212/search', model_path='search')
    session = Session()
    client._http_session = session
    result = asyncio.run(client.execute(query='example', num_results=2))
    assert session.url == 'http://gateway:1212/search'
    assert session.kwargs['json'] == {
        'mode': 'query', 'query': 'example', 'num_results': 2,
        'parameters': {}, 'model_path': 'search',
    }
    assert result == {'results': [{'url': 'https://example.org'}]}


def test_web_assets_normalize_fetch_envelopes_and_own_their_data():
    store = WebAssetStore({'https://example.org': json.dumps({
        'mode': 'url', 'data': {'content': 'page body'},
    })})
    asset_id = store.get_id_by_url('https://example.org')
    assert store.get(asset_id)['content'] == 'page body'
    returned = store.get(asset_id)
    returned['content'] = 'changed'
    assert store.get(asset_id)['content'] == 'page body'
    assert store.add_url('https://example.org') == asset_id
