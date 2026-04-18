# config.py – Constants and skill keywords for the AI CV Screening Tool

APP_TITLE = "AI CV Screening Tool"
APP_ICON = "🤖"
APP_DESCRIPTION = (
    "Upload a Job Description and multiple CV PDFs. "
    "The tool uses multilingual AI (sentence-transformers) to rank, highlight, "
    "and find skill gaps in each candidate — supports Vietnamese & English."
)

# Multilingual model: supports 50+ languages (Vietnamese + English cross-lingual)
# ~470 MB, runs on CPU, no API key needed
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# How many text chunks to show as evidence per CV
TOP_K_EVIDENCE = 3

# Chunk size (characters) when splitting CV text
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Score thresholds for colour-coding
SCORE_HIGH = 0.70   # green  ≥ 70 %
SCORE_MED  = 0.50   # yellow ≥ 50 %
# below SCORE_MED  → red

# Composite score weights
WEIGHT_SEMANTIC    = 0.65   # semantic similarity weight
WEIGHT_SKILL       = 0.35   # skill coverage weight
WEIGHT_EXPERIENCE  = 0.0    # experience score weight (0 = disabled by default)

# Experience parsing / scoring
MAX_EXPERIENCE_YEARS          = 40    # sanity cap on detected years
EXPERIENCE_NORMALIZATION_YEARS = 5.0  # full score (1.0) at this many years

# Red flag detection thresholds
MIN_CV_LENGTH              = 300    # chars below which CV is flagged as too short
MAX_SKILLS_WITHOUT_PROJECT = 15     # skill count above which no-project is a red flag
MAX_SKILL_DENSITY          = 0.05   # skill_count/word_count ratio flagged as abnormal

# Insight dashboard thresholds
MIN_CANDIDATES_FOR_INTERVIEW    = 5   # ≥ this many high-match → suggest interviews
MIN_CANDIDATES_FOR_LOW_WARNING  = 5   # need at least this many CVs to warn on low %
LOW_MATCH_PCT_THRESHOLD         = 20  # if high-match% < this → suggest lowering bar
SKILL_GAP_WARNING_THRESHOLD     = 60  # if X% of CVs missing a skill → warn HR

# SQLite database path (relative to app root)
DATABASE_PATH = "cv_screening.db"

# ---------------------------------------------------------------------------
# Skill keyword catalogue (extend freely – includes Vietnamese aliases)
# ---------------------------------------------------------------------------
SKILL_KEYWORDS: dict[str, list[str]] = {
    # Programming languages
    "Python":       ["python", "py", "django", "flask", "fastapi", "pandas", "numpy",
                     "lập trình python", "ngôn ngữ python"],
    "JavaScript":   ["javascript", "js", "es6", "typescript", "ts", "nodejs", "node.js",
                     "lập trình javascript"],
    "Java":         ["java", "spring", "springboot", "maven", "gradle", "lập trình java"],
    "C++":          ["c++", "cpp", "stl"],
    "Go":           ["golang", "go "],
    "Rust":         ["rust", "cargo"],
    "PHP":          ["php", "laravel", "symfony"],
    "Ruby":         ["ruby", "rails"],
    "Swift":        ["swift", "xcode", "ios"],
    "Kotlin":       ["kotlin", "android"],

    # Front-end
    "React":        ["react", "reactjs", "react.js", "redux", "next.js", "nextjs"],
    "Vue":          ["vue", "vuejs", "vue.js", "nuxt"],
    "Angular":      ["angular", "angularjs"],
    "HTML/CSS":     ["html", "css", "sass", "scss", "tailwind", "bootstrap", "giao diện web"],

    # Back-end / infra
    "PostgreSQL":   ["postgresql", "postgres", "psql"],
    "MySQL":        ["mysql", "mariadb", "cơ sở dữ liệu mysql"],
    "MongoDB":      ["mongodb", "mongo", "mongoose"],
    "Redis":        ["redis"],
    "Docker":       ["docker", "dockerfile", "docker-compose", "containeriz", "container"],
    "Kubernetes":   ["kubernetes", "k8s", "helm"],
    "AWS":          ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "điện toán đám mây aws"],
    "GCP":          ["gcp", "google cloud", "bigquery", "cloud run"],
    "Azure":        ["azure", "microsoft azure"],
    "CI/CD":        ["ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "travis",
                     "tích hợp liên tục", "triển khai liên tục"],
    "Linux":        ["linux", "unix", "bash", "shell script", "hệ điều hành linux"],

    # Data / AI / ML
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "sklearn", "xgboost", "lightgbm",
                         "học máy", "máy học", "mô hình ml"],
    "Deep Learning":    ["deep learning", "tensorflow", "keras", "pytorch", "neural network",
                         "học sâu", "mạng nơ-ron", "mạng neural"],
    "NLP":              ["nlp", "natural language", "huggingface", "transformers", "spacy", "nltk",
                         "xử lý ngôn ngữ tự nhiên", "nlp tiếng việt"],
    "Data Analysis":    ["data analysis", "data analytics", "tableau", "power bi", "excel", "sql",
                         "phân tích dữ liệu", "phân tích số liệu"],
    "SQL":              ["sql", "query", "database", "orm", "truy vấn", "cơ sở dữ liệu"],
    "Computer Vision":  ["computer vision", "opencv", "image processing", "object detection", "yolo",
                         "thị giác máy tính", "xử lý ảnh"],

    # Soft / process
    "Agile":            ["agile", "scrum", "kanban", "jira", "sprint",
                         "phương pháp agile", "làm việc nhóm agile"],
    "Git":              ["git", "github", "gitlab", "bitbucket", "version control",
                         "quản lý mã nguồn", "quản lý phiên bản"],
    "REST API":         ["rest api", "restful", "api design", "openapi", "swagger",
                         "thiết kế api", "xây dựng api"],
    "GraphQL":          ["graphql"],
    "Microservices":    ["microservice", "microservices", "service mesh", "kiến trúc microservice"],
    "Testing":          ["unit test", "pytest", "jest", "cypress", "tdd", "bdd", "selenium",
                         "kiểm thử", "viết test", "automated testing"],

    # Vietnamese-specific roles / soft skills
    "Giao tiếp":        ["giao tiếp tốt", "kỹ năng giao tiếp", "communication", "thuyết trình"],
    "Quản lý dự án":    ["quản lý dự án", "project management", "pm", "pmp", "lập kế hoạch dự án"],
    "Làm việc nhóm":    ["làm việc nhóm", "teamwork", "team player", "phối hợp nhóm"],
    "Tư duy phân tích": ["tư duy phân tích", "analytical thinking", "problem solving", "giải quyết vấn đề"],

    # AI / LLM / Modern ML
    "LangChain":        ["langchain", "lang chain", "langchain4j"],
    "LlamaIndex":       ["llamaindex", "llama index", "llama_index"],
    "OpenAI API":       ["openai", "chatgpt api", "gpt-4", "gpt-3.5", "davinci", "gpt4", "gpt3"],
    "Hugging Face":     ["huggingface", "hugging face", "transformers", "datasets", "peft", "trl"],
    "MLflow":           ["mlflow", "ml flow"],
    "Airflow":          ["airflow", "apache airflow", "dag", "workflow orchestration"],
    "Spark":            ["spark", "apache spark", "pyspark", "databricks", "rdd"],

    # Event streaming / messaging
    "Kafka":            ["kafka", "apache kafka", "confluent", "event streaming",
                         "message queue kafka"],
    "RabbitMQ":         ["rabbitmq", "rabbit mq", "amqp", "message broker"],
    "Celery":           ["celery", "celery worker", "task queue"],

    # Search & observability
    "Elasticsearch":    ["elasticsearch", "elastic search", "opensearch", "kibana",
                         "elk stack", "logstash"],
    "Prometheus":       ["prometheus", "grafana", "alertmanager", "observability metrics"],

    # Infrastructure / DevOps
    "Terraform":        ["terraform", "iac", "infrastructure as code", "hcl"],
    "Ansible":          ["ansible", "playbook", "configuration management"],
    "Nginx":            ["nginx", "reverse proxy", "web server nginx"],

    # Mobile
    "Flutter":          ["flutter", "dart", "flutter developer"],
    "React Native":     ["react native", "reactnative", "expo", "mobile react"],

    # Modern back-end / protocols
    "FastAPI":          ["fastapi", "fast api"],
    "gRPC":             ["grpc", "protobuf", "protocol buffer"],
    "WebSocket":        ["websocket", "socket.io", "real-time websocket"],
    "Next.js":          ["next.js", "nextjs", "ssr react", "server side rendering react"],
    "TypeScript":       ["typescript", "ts strict", "type safe", "typed javascript"],
}