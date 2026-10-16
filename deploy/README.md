# Deploying Femida

The frontend image is built by GitHub Actions on every push and tagged
`ghcr.io/emsch/femida:<sha>` (`.github/workflows/build.yml`). Deploys are
manual: `Deploy to dev stand` and `Deploy to production` take a full commit sha
and ssh into German, where `femida-deploy.sh` (installed as
`/usr/local/sbin/femida-deploy`) moves the checkout to that sha, rebuilds the
frontend image from it and brings the stack up, rolling back to the previous
image if `/login` stops answering.

Each workflow has its own ed25519 key. On German every key is pinned in
`/root/.ssh/authorized_keys` to one target and nothing else:

```
restrict,command="/usr/local/sbin/femida-deploy dev"  ssh-ed25519 ... gha-femida-deploy-dev
restrict,command="/usr/local/sbin/femida-deploy prod" ssh-ed25519 ... gha-femida-deploy-prod
```

The dev key lives in the repository secret `GERMAN_DEV_DEPLOY_SSH_KEY`. The
prod key lives in the `production` environment as `GERMAN_PROD_DEPLOY_SSH_KEY`;
that environment requires a reviewer, so a production deploy waits for an
approval before it can even read the key.

| target | directory | compose file | address |
|---|---|---|---|
| dev | `/femida-dev` | `docker-compose.dev.yml` | `10.0.0.2:8092`, `femida-dev.emsch.ru` via IKELOS |
| prod | `/femida` | `docker-compose.yaml` | `10.0.0.2:8082`, `femida.emsch.ru` via IKELOS |

The dev stand holds synthetic data only (`local/seed`) and signs you in
without Google through `FEMIDA_DEV_LOGIN` (`local/envfile.local`).
