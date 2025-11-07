from chromadb import Client
from chromadb.config import Settings

# 로컬에 저장된 chroma DB 연결
client = Client(Settings(
    chroma_db_impl="chromadb.db.duckdb.DuckDB",
    persist_directory="./chroma_creation"  # 네 폴더명에 맞게
))

# 현재 저장된 컬렉션 리스트 확인
print("📂 현재 컬렉션 목록:")
collections = client.list_collections()
for col in collections:
    print("-", col.name)

# 예시로 첫 번째 컬렉션 내용 확인
if collections:
    collection = client.get_collection(collections[0].name)
    print(f"\n📄 '{collections[0].name}' 컬렉션 내용 미리보기:")
    data = collection.get(limit=3)  # 처음 3개만 미리보기
    print(data)
else:
    print("❌ 컬렉션이 없습니다. build_index.py에서 잘 저장됐는지 확인하세요.")
