"""Continue a video with its actual 22-frame prefix included in multimodal text conditioning."""
from dense_stream import main
if __name__ == '__main__':
    main(visual_context=True)
