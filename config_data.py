from pathlib import Path

# The repository once committed a SQLite database and an MD5 manifest without
# Chroma's segment files. Keep runtime state separate so a fresh deployment
# rebuilds a complete index instead of trusting that partial snapshot.
_RUNTIME_DIR = Path(__file__).resolve().parent / ".runtime"
md5_path = str(_RUNTIME_DIR / "md5.text")

# Chroma
collection_name = "rag"
persist_directory = str(_RUNTIME_DIR / "chroma_db")

# splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n","。","！","？",".","!","?"," ",""]
max_split_char_number = 1000

similarity_threshold = 2    # 检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen4-max"

session_config = {
    "configurable":{
        "session_id":"user_001"
    }
}
