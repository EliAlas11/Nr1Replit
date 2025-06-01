
<old_str># nr1copilot

A modern AI-powered tool to transform YouTube videos into viral clips for TikTok, Instagram Reels, and YouTube Shorts.

## Features

- Convert YouTube videos to short, viral-ready clips
- AI-powered editing and optimization
- Download and process videos with ease
- Modern, responsive web interface

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm

### Installation

```bash
git clone https://github.com/EliAlas11/nr1copilot.git
cd nr1copilot/nr1-main
npm install
```

### Running the App

```bash
npm start
```

The server will start on `http://localhost:5000` by default.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)</old_str>
<new_str># 🎬 Viral Clip Generator v2.0

A professional-grade, AI-powered platform for transforming YouTube videos into viral clips optimized for TikTok, Instagram Reels, and YouTube Shorts.

## ✨ Features

### 🎯 Core Functionality
- **YouTube Integration**: Direct video processing from YouTube URLs
- **Smart Clipping**: AI-powered identification of viral-worthy moments
- **Multi-Platform Export**: Optimized outputs for TikTok, Instagram, YouTube Shorts
- **Real-time Processing**: Live progress tracking with detailed status updates
- **Quality Options**: Multiple resolution outputs (360p, 720p, 1080p)

### 🎨 User Experience
- **Modern UI**: Responsive design with dark/light mode support
- **Progressive Web App**: Offline support and mobile-optimized
- **Accessibility**: WCAG 2.1 AA compliant with keyboard navigation
- **Real-time Feedback**: Toast notifications and progress indicators
- **Batch Processing**: Queue multiple videos for processing

### 🛠️ Technical Excellence
- **FastAPI Backend**: High-performance async Python API
- **Production Ready**: Comprehensive logging, error handling, monitoring
- **Scalable Architecture**: Microservices-ready with Redis/MongoDB
- **Security First**: JWT authentication, rate limiting, input validation
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- FFmpeg
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/EliAlas11/nr1copilot.git
cd nr1copilot/nr1-main
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the application**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

5. **Access the application**
- Web Interface: http://localhost:5000
- API Documentation: http://localhost:5000/docs
- Health Check: http://localhost:5000/health

## 📚 API Documentation

### Core Endpoints

#### Video Processing
```http
POST /api/v1/video/validate
POST /api/v1/video/download
POST /api/v1/video/process
GET  /api/v1/videos
```

#### Authentication
```http
POST /api/v1/auth/signup
POST /api/v1/auth/login
POST /api/v1/auth/refresh
```

#### User Management
```http
GET  /api/v1/user/profile
PUT  /api/v1/user/profile
POST /api/v1/user/change-password
```

### Example Usage

**Validate YouTube URL:**
```bash
curl -X POST "http://localhost:5000/api/v1/video/validate" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Process Video Clip:**
```bash
curl -X POST "http://localhost:5000/api/v1/video/process" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "start_time": 30.0,
    "end_time": 90.0,
    "quality": "720p"
  }'
```

## 🏗️ Architecture

### Backend Structure
```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── schemas.py           # Pydantic models
├── routes/              # API route handlers
├── services/            # Business logic layer
├── models/              # Database models
├── utils/               # Utility functions
└── tests/               # Comprehensive test suite
```

### Key Components
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **yt-dlp**: YouTube video downloading and metadata extraction
- **FFmpeg**: Video processing and format conversion
- **MongoDB**: Document database for user data and video metadata
- **Redis**: Caching and job queue management
- **Celery**: Distributed task processing

## 🔧 Configuration

### Environment Variables
```bash
# Application
DEBUG=false
ENVIRONMENT=production
HOST=0.0.0.0
PORT=5000

# Database
MONGODB_URI=mongodb://localhost:27017/viral_clips
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your-jwt-secret-here
SECRET_KEY=your-secret-key-here

# Processing Limits
MAX_VIDEO_DURATION=300     # 5 minutes
MAX_FILE_SIZE=104857600    # 100MB
```

### Production Deployment

**Using Docker:**
```bash
docker build -t viral-clip-generator .
docker run -p 5000:5000 viral-clip-generator
```

**Using Systemd:**
```bash
sudo cp viral-clip-generator.service /etc/systemd/system/
sudo systemctl enable viral-clip-generator
sudo systemctl start viral-clip-generator
```

## 🧪 Testing

### Run Test Suite
```bash
# Unit tests
pytest app/tests/

# With coverage
pytest --cov=app app/tests/

# Specific test file
pytest app/tests/test_video_service.py -v
```

### API Testing
```bash
# Health check
curl http://localhost:5000/health

# Load test with artillery
artillery quick --count 10 --num 5 http://localhost:5000/health
```

## 🔒 Security Features

- **Input Validation**: Comprehensive request/response validation
- **Rate Limiting**: Configurable per-endpoint rate limits
- **JWT Authentication**: Secure token-based authentication
- **CORS Protection**: Configurable cross-origin request handling
- **SQL Injection Prevention**: Parameterized database queries
- **XSS Protection**: Content Security Policy headers

## 📊 Monitoring & Logging

### Health Monitoring
- Application health: `/health`
- Dependencies check: `/health/dependencies`
- Metrics endpoint: `/metrics` (Prometheus compatible)

### Logging Levels
- **INFO**: General application flow
- **WARNING**: Unusual but recoverable events
- **ERROR**: Error conditions that need attention
- **DEBUG**: Detailed diagnostic information

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black app/
isort app/
flake8 app/
```

### Code Style
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloader
- [FFmpeg](https://ffmpeg.org/) - Video processing toolkit

## 📞 Support

- **Documentation**: [docs.viralclipgenerator.com](https://docs.viralclipgenerator.com)
- **Issues**: [GitHub Issues](https://github.com/EliAlas11/nr1copilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EliAlas11/nr1copilot/discussions)

---

Built with ❤️ using FastAPI, Python, and modern web technologies.</new_str>
