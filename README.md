# TFG - Adrián Racero Serrano

## Detección de imágenes generadas por Inteligencia Artificial mediante la Ley de Benford

Este repositorio contiene el desarrollo del Trabajo de Fin de Grado (TFG). La estructura del proyecto está organizada para facilitar el desarrollo teórico, la implementación de código y la visualización a través de una aplicación.

## Estructura del Proyecto

tfg/

├── notebooks/   # Desarrollo teórico 

├── code/        # Código fuente del proyecto

├── app/         # Aplicación 


## Levantar el entorno con Docker Compose

El entorno de desarrollo está completamente preparado para ejecutarse dentro de un contenedor Docker con soporte para GPU (NVIDIA). Asegúrate de tener instalado:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (para usar la GPU)

### Comandos

> Los pasos 1 y 2 solo son necesarios la **primera vez**.

**1. Clonar el repositorio:**

```bash
git clone https://github.com/AdrianRacero/benford-art.git
cd benford-art
```

**2. Construir la imagen del contenedor:**

```bash
docker-compose build
```

**3. Levantar el servicio:**

```bash
docker-compose up
```

**4. Acceder a JupyterLab:** abre tu navegador y entra a:

```bash
http://localhost:8888
```

---

## Detection of Artificial Intelligence generated images using Benford's Law

This repository contains the development of the Final Degree Project (TFG). The project structure is organized to support theoretical development, code implementation, and visualization through an application.

## Project Structure

tfg/

├── notebooks/   # Theoretical development

├── code/        # Project source code

├── app/         # Application

## Launching the Environment with Docker Compose

The development environment is fully configured to run inside a Docker container with GPU (NVIDIA) support. Make sure you have installed:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Commands

> Steps 1 and 2 are only required the **first time**.

**1. Clone the repository:**

```bash
git clone https://github.com/AdrianRacero/benford-art.git
cd benford-art
```

**2. Build the container image:**

```bash
docker-compose build
```

**3. Start the service:**

```bash
docker-compose up
```

**4. Access JupyterLab:** open your browser and go to:

```bash
http://localhost:8888
```
