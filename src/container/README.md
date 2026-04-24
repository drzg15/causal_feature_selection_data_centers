# Container Setup

## Starting the Container

To start the container services, navigate to the container directory and run the following command:

CPU only
```bash
docker compose up
```

GPU support
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

This will build and start the following services:
- nf-nhits (port 8020)
- nf-tsmixerx (port 8021)
- nf-patchtst (port 8022)

To run in detached mode (background), use:

CPU only
```bash
docker compose up -d
```

GPU support
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

To stop the services:

```bash
docker compose down
```