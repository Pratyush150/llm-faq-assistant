"""HTTP API, chat UI and CLI. Loopback only, no external network."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from faqbot.cli import build_parser, main
from faqbot.pipeline import FAQPipeline, PipelineConfig
from faqbot.server import CHAT_HTML, make_handler


@pytest.fixture(scope="module")
def server(request):
    corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus")
    goldset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "goldset.json")
    if not os.path.isdir(corpus):
        pytest.skip("bundled corpus is not present")
    pipe = FAQPipeline(PipelineConfig(label="server"))
    pipe.ingest([corpus])
    handler = make_handler(pipe, goldset_path=goldset, quiet=True)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError:  # pragma: no cover - sandbox without loopback
        pytest.skip("cannot bind a loopback socket here")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    yield base
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


def _post(base: str, path: str, payload: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_root_serves_the_chat_ui(server):
    status, body = _get(server, "/")
    assert status == 200
    assert "<title>faqbot</title>" in body
    assert "/ask" in body


def test_health_reports_the_index(server):
    status, body = _get(server, "/health")
    payload = json.loads(body)
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["chunks"] > 40
    assert payload["embedder"] == "hashing"
    assert payload["embedder_capabilities"]["hashing"]["available"] is True


def test_ask_returns_a_cited_answer(server):
    status, payload = _post(server, "/ask", {"question": "How long does the AR-1 battery last?"})
    assert status == 200
    assert payload["refused"] is False
    assert "90 minutes" in payload["text"]
    assert payload["citations"]
    assert payload["chunk_ids"]
    assert payload["groundedness"] == 1.0


def test_ask_refuses_out_of_domain(server):
    _, payload = _post(server, "/ask", {"question": "What is the best sourdough bread recipe?"})
    assert payload["refused"] is True
    assert payload["refusal_reason"] == "out_of_domain"


def test_ask_keeps_session_context(server):
    _post(server, "/ask", {"question": "How long does the AR-1 battery last?", "session_id": "s1"})
    _, payload = _post(server, "/ask", {"question": "Does it work on carpet?", "session_id": "s1"})
    assert payload["rewritten_question"] == "Does the AR-1 work on carpet?"


def test_ask_validates_its_input(server):
    status, payload = _post(server, "/ask", {})
    assert status == 400
    assert "question" in payload["error"]


def test_ingest_rejects_a_missing_path(server):
    status, payload = _post(server, "/ingest", {"paths": ["/no/such/place"]})
    assert status == 400
    assert "no such path" in payload["error"]


def test_eval_endpoint_returns_metrics(server):
    status, body = _get(server, "/eval")
    payload = json.loads(body)
    assert status == 200
    assert payload["questions"] >= 25
    assert "recall_at" in payload
    assert payload["correct_refusals"] >= 0.0


def test_unknown_route_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/nope")
    assert exc.value.code == 404


def test_chat_html_is_self_contained():
    """No CDN, no build step: the UI must work with the network switched off."""
    assert "<script src=" not in CHAT_HTML
    assert "http://" not in CHAT_HTML.replace("http://127.0.0.1", "")
    assert "cdn" not in CHAT_HTML.casefold()


# -- CLI ---------------------------------------------------------------------


def test_parser_exposes_every_command():
    parser = build_parser()
    args = parser.parse_args(["ask", "How long?"])
    assert args.command == "ask"
    assert args.question == "How long?"
    assert parser.parse_args(["--demo"]).demo is True
    assert parser.parse_args(["eval", "--compare"]).compare is True


def test_cli_ask_prints_json(capsys, corpus_dir):
    code = main(["ask", "How long does the AR-1 battery last?", "--corpus", corpus_dir, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["refused"] is False
    assert "90 minutes" in payload["text"]


def test_cli_ask_returns_a_nonzero_code_on_refusal(capsys, corpus_dir):
    code = main(["ask", "What is the best sourdough recipe?", "--corpus", corpus_dir])
    capsys.readouterr()
    assert code == 2


def test_cli_ingest_and_reload_an_index(capsys, corpus_dir, tmp_path):
    index_path = os.path.join(str(tmp_path), "index.json")
    assert main(["ingest", corpus_dir, "--save", index_path]) == 0
    assert "indexed" in capsys.readouterr().out
    assert os.path.exists(index_path)
    assert main(["ask", "Which filter part number does the AR-1 use?", "--index", index_path]) == 0
    assert "NW-FILT-02" in capsys.readouterr().out


def test_cli_eval_prints_a_report(capsys):
    from conftest import GOLDSET_PATH

    if not os.path.exists(GOLDSET_PATH):
        pytest.skip("bundled goldset is not present")
    assert main(["eval", GOLDSET_PATH]) == 0
    out = capsys.readouterr().out
    assert "recall@1" in out
    assert "correct refusals" in out


def test_cli_capabilities_lists_the_offline_default(capsys):
    assert main(["capabilities"]) == 0
    out = capsys.readouterr().out
    assert "hashing" in out
    assert "offline" in out


def test_cli_with_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.casefold()
