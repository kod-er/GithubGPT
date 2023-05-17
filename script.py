import pickle
import os
import nest_asyncio

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
os.environ['GITHUB_TOKEN'] = 'YOUR_GITHUB_TOKEN'

from llama_index import download_loader, GPTVectorStoreIndex
from llama_index.readers.llamahub_modules.github_repo import GithubRepositoryReader, GithubClient


print(GithubRepositoryReader)
docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "OWNER",
        repo =                   "REPO",
        filter_directories =     (["src"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".js", ".tsx"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTVectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()
print(query_engine.query("YOUR_QUERY"))
