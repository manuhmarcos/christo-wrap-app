# Christo Wrap App

Aplicacion mobile para envolver objetos fotografiados al estilo de Christo y Jeanne-Claude.

---

## Estructura del proyecto

```
christo-app/
├── backend/        ← Servidor Python (se sube a Railway)
└── app/            ← App Flutter (se instala en el celular)
```

---

## PASO 1 — Configurar y deployar el backend

### 1.1 Crear el archivo .env

Dentro de la carpeta `backend/`, crea un archivo llamado `.env` (copia de `.env.example`):

```
REPLICATE_API_TOKEN=r8_TuTokenReal
```

Tu token lo encontras en: https://replicate.com/account/api-tokens

### 1.2 Subir a Railway

1. Ir a https://railway.app → New Project → Deploy from GitHub
2. Conectar tu repositorio de GitHub (primero subi este proyecto a GitHub)
3. Seleccionar la carpeta `backend/` como root directory
4. En la seccion **Variables**, agregar:
   - `REPLICATE_API_TOKEN` = tu token de Replicate
5. Railway va a detectar el `Procfile` y deployar automaticamente
6. Una vez deployado, copiar la URL publica (ej: `https://christo-app-production.up.railway.app`)

### 1.3 Actualizar la URL en la app

Abrir `app/lib/services/api_service.dart` y reemplazar:
```dart
static const String baseUrl = 'https://TU-BACKEND.up.railway.app';
```
por la URL real de tu backend.

---

## PASO 2 — Configurar y compilar la app Flutter

### 2.1 Instalar Flutter

Seguir las instrucciones oficiales: https://flutter.dev/docs/get-started/install

### 2.2 Instalar dependencias

```bash
cd app/
flutter pub get
```

### 2.3 Configurar permisos iOS

En `app/ios/Runner/Info.plist`, agregar las claves del archivo `Info.plist.additions`.

### 2.4 Configurar permisos Android

En `app/android/app/src/main/AndroidManifest.xml`, agregar los permisos del archivo `AndroidManifest.additions.xml`.

### 2.5 Correr en el celular

```bash
flutter run
```

---

## Como funciona la IA

1. **SAM (Segment Anything Model)** de Meta detecta automaticamente el objeto principal de la foto
2. La mascara del objeto se dilata ligeramente para mejor cobertura
3. **Stable Diffusion XL Inpainting** genera la tela envolviendo el objeto de forma fotorrealista
4. El resultado se devuelve a la app en segundos

---

## Materiales disponibles

| Material | Descripcion |
|----------|-------------|
| Tela     | Lino blanco, estilo clasico Christo |
| Plastico | Polipropileno brillante con cuerdas |
