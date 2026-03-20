-- Run this script in your target PostgreSQL database.
-- If you use a different embedding model dimension, adjust vector(768) accordingly.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.langchain_pg_collection (
	uuid uuid NOT NULL,
	name varchar,
	cmetadata json,
	CONSTRAINT langchain_pg_collection_pkey PRIMARY KEY (uuid)
);

CREATE TABLE IF NOT EXISTS public.langchain_pg_embedding (
	uuid uuid NOT NULL,
	collection_id uuid,
	embedding vector(768),
	document varchar,
	cmetadata json,
	custom_id varchar,
	CONSTRAINT langchain_pg_embedding_pkey PRIMARY KEY (uuid)
);

CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_collection_id
	ON public.langchain_pg_embedding (collection_id);

CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_embedding_cosine
	ON public.langchain_pg_embedding
	USING ivfflat (embedding vector_cosine_ops)
	WITH (lists = 100);
