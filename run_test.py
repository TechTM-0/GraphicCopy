import sys
import io
sys.path.insert(0, 'src')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from graphiccopy.ocr import load_image, run_ocr_with_preprocess
import time

for path in ['samples/フロー.png', 'samples/test_flowchart.png']:
    print(f'\n=== {path} ===')
    img = load_image(path)
    print(f'Size: {img.shape[1]}x{img.shape[0]}')
    t = time.time()
    result = run_ocr_with_preprocess(img, verbose=False)
    elapsed = time.time() - t
    blocks = result['text_blocks']
    print(f'Time: {elapsed:.1f}s  Blocks: {len(blocks)}')
    for b in blocks:
        print(f'  [{b["confidence"]:5.1f}] {b["text"]}  @{b["bbox"]}')
