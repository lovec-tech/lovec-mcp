# lovec-mcp

Локальный MCP-сервер поверх детектора промпт-инъекций [lovec.tech](https://lovec.tech).
Даёт агентам (Claude Desktop, Claude Code, любой MCP-клиент) тул `check_prompt_injection` —
проверка недоверенного текста (веб-страница, документ, результат тула, письмо) перед тем,
как отдать его в другую LLM.

Работает **только с вашим собственным ключом** — сервер ничего не хранит и не шарит,
это тонкий клиент поверх уже существующего API-ключа/баланса с сайта.

## Установка

```bash
cd lovec-mcp
python3 -m venv .venv
./.venv/bin/pip install -e .
```

Ключ выпускается на [lovec.tech](https://lovec.tech)

## Быстрая проверка руками

```bash
export LOVEC_KEY=aig_...
./.venv/bin/python server.py
```

```bash
export LOVEC_KEY=aig_...
./.venv/bin/python -c "
import asyncio, server
print(asyncio.run(server.check_prompt_injection('тестовый текст')))
"
```

## Подключение к MCP-клиенту

Claude Desktop (`claude_desktop_config.json`) или Claude Code (`.mcp.json`) — один и тот же формат:

```json
{
  "mcpServers": {
    "lovec": {
      "command": "/absolute/path/to/lovec-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/lovec-mcp/server.py"],
      "env": { "LOVEC_KEY": "aig_..." }
    }
  }
}
```

## Переменные окружения

| Переменная | По умолчанию | Зачем |
|---|---|---|
| `LOVEC_KEY` | — (обязательна) | ключ с lovec.tech |
| `LOVEC_BASE` | `https://lovec.tech` | другой хост API |
| `LOVEC_TIMEOUT` | `60` | потолок ожидания одного вызова, секунды |

