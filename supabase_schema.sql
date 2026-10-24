-- Run this in your Supabase SQL Editor:
-- https://supabase.com/dashboard/project/plaiursharanizyrdtmt/sql

CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    kind TEXT NOT NULL,                -- 'ask' or 'soundscape'
    prompt_or_query TEXT NOT NULL,     -- question text or uploaded image filename
    result_summary TEXT NOT NULL,      -- answer excerpt or soundscape concept
    vision_model TEXT,                 -- e.g. 'gemma-4-26b-a4b-it'
    audio_model TEXT,                  -- e.g. 'gemini-3.8-flash-tts' or 'meta/muse-glimmer-30b'
    attempts INT DEFAULT 1,
    has_image BOOLEAN DEFAULT FALSE
);

-- Enable Row Level Security (RLS) if you want public inserts/selects:
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public inserts" ON interactions
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public reads" ON interactions
    FOR SELECT USING (true);

