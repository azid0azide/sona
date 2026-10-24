# Sample inputs

Keep three to five representative inputs here: the ones you used in Colab while
getting the prompt right, plus at least one bad case (blurry photo, no text in
the image, a question unrelated to the image).

These are not run by pytest, which stays offline. They are for you:
after changing `prompts/system.txt`, run each one through the app and check the
answer still has the shape the prompt promises.

    python -m scripts.smoke tests/sample_inputs/your-photo.jpg

Don't commit images you don't have the right to share. The repo is public.
