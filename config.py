# config.py – Constants and skill keywords for the AI CV Screening Tool

APP_TITLE = "AI CV Screening Tool"
APP_ICON = "🤖"
APP_DESCRIPTION = (
    "Upload a Job Description and multiple CV PDFs. "
    "The tool uses semantic AI (sentence-transformers) to rank, highlight, "
    "and find skill gaps in each candidate."
)

# sentence-transformers model – 33 MB, runs on CPU, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many text chunks to show as evidence per CV
TOP_K_EVIDENCE = 3

# Chunk size (characters) when splitting CV text
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Score thresholds for colour-coding
SCORE_HIGH = 0.70   # green  ≥ 70 %
SCORE_MED  = 0.50   # yellow ≥ 50 %
# below SCORE_MED  → red

# ---------------------------------------------------------------------------
# Skill keyword catalogue (extend freely)
# ---------------------------------------------------------------------------
SKILL_KEYWORDS: dict[str, list[str]] = {
    # Programming languages
    "Python":       ["python", "py", "django", "flask", "fastapi", "pandas", "numpy"],
    "JavaScript":   ["javascript", "js", "es6", "typescript", "ts", "nodejs", "node.js"],
    "Java":         ["java", "spring", "springboot", "maven", "gradle"],
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
    "HTML/CSS":     ["html", "css", "sass", "scss", "tailwind", "bootstrap"],

    # Back-end / infra
    "PostgreSQL":   ["postgresql", "postgres", "psql"],
    "MySQL":        ["mysql", "mariadb"],
    "MongoDB":      ["mongodb", "mongo", "mongoose"],
    "Redis":        ["redis"],
    "Docker":       ["docker", "dockerfile", "docker-compose", "containeriz"],
    "Kubernetes":   ["kubernetes", "k8s", "helm"],
    "AWS":          ["aws", "amazon web services", "ec2", "s3", "lambda", "rds"],
    "GCP":          ["gcp", "google cloud", "bigquery", "cloud run"],
    "Azure":        ["azure", "microsoft azure"],
    "CI/CD":        ["ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "travis"],
    "Linux":        ["linux", "unix", "bash", "shell script"],

    # Data / AI / ML
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "sklearn", "xgboost", "lightgbm"],
    "Deep Learning":    ["deep learning", "tensorflow", "keras", "pytorch", "neural network"],
    "NLP":              ["nlp", "natural language", "huggingface", "transformers", "spacy", "nltk"],
    "Data Analysis":    ["data analysis", "data analytics", "tableau", "power bi", "excel", "sql"],
    "SQL":              ["sql", "query", "database", "orm"],

    # Soft / process
    "Agile":            ["agile", "scrum", "kanban", "jira", "sprint"],
    "Git":              ["git", "github", "gitlab", "bitbucket", "version control"],
    "REST API":         ["rest api", "restful", "api design", "openapi", "swagger"],
    "GraphQL":          ["graphql"],
    "Microservices":    ["microservice", "microservices", "service mesh"],
    "Testing":          ["unit test", "pytest", "jest", "cypress", "tdd", "bdd", "selenium"],
}
