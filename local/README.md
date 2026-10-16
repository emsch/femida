# Локальный стенд Фемиды

Поднимает фронтенд, MongoDB и синтетические данные без единого реального
секрета. Прод-файл `docker-compose.yaml` не используется.

```
docker compose -f docker-compose.local.yml up --build
```

Затем открыть <http://127.0.0.1:8082/login> — вход происходит без Google под
пользователем из `FEMIDA_DEV_LOGIN` (см. `local/envfile.local` и
`frontend/app/dev_login.py`). Страницы: `/`, `/pdf.html`, `/monitor.html`,
`/leaderboard.html`, выгрузка `/export`.

Сервис `seed` один раз наполняет базу вымышленными работами и кладёт
SVG-заглушки вместо сканов в `local/media/ocr_results/` (каталог в
`.gitignore`). Повторный запуск ничего не дублирует. Сброс данных:

```
docker compose -f docker-compose.local.yml down -v
```

mongo-express по желанию: `--profile tools`, <http://127.0.0.1:8081>,
логин и пароль `femida`.

Образ фронтенда собирается под `linux/amd64` — базовый образ
`python3.6-alpine3.7` существует только для этой архитектуры.
