import chromadb
import pandas as pd
import streamlit as st

pd.set_option("display.max_columns", 4)


def view_collections(db_path):
    st.markdown(f"### DB Path: 📁 `{db_path}`")
    client = chromadb.PersistentClient(path=db_path)

    st.header("📚 Available Collections")
    collection_names = client.list_collections()
    st.write(f"**Total Collections:** {len(collection_names)}")

    for i, name in enumerate(collection_names, 1):
        try:
            collection = client.get_collection(name=name)
            metadata = collection.metadata

            st.markdown(f"---\n### {i}. Collection: **{name}**")
            st.write("📄 **Metadata:**", metadata)

            # Try to get collection data
            data = collection.get()

            ids = data.get("ids") or []
            documents = data.get("documents") or []
            embeddings = data.get("embeddings") or []  # noqa: F841
            metadatas = data.get("metadatas") or []

            min_len = min(len(ids), len(documents), len(metadatas)) if ids else 0
            st.text(f"🔢 Total Items: {len(ids)} | Displaying: {min_len} sample rows")

            # Build DataFrame
            df = pd.DataFrame(
                {
                    "ID": ids[:min_len],
                    "Document": documents[:min_len],
                    "Metadata": metadatas[:min_len],
                }
            )

            st.dataframe(df)

        except Exception as e:
            st.warning(f"⚠️ Could not load collection '{name}': {e}")


if __name__ == "__main__":
    st.title("🧠 ChromaDB Viewer")

    db_path = st.text_input(
        "Enter Chroma DB Path:",
        value="./chromadb_store",
        help="Paste your Chroma DB directory path here",
    )

    if db_path:
        try:
            view_collections(db_path)
        except Exception as err:
            st.error(f"Failed to view collections: {err}")
