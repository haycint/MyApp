from pathlib import Path

replacements = [
    ('annotation/annotation_dataset.new.py', 'annotation/annotation_dataset.py'),
    ('annotation/annotation_model.new.py', 'annotation/annotation_model.py'),
    ('annotation/train_annotation.new.py', 'annotation/train_annotation.py'),
    ('annotation/annotation_tool.new.py', 'annotation/annotation_tool.py')
]

for src, dst in replacements:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.exists():
        dst_path.unlink(missing_ok=True)
        src_path.replace(dst_path)
        print(f'Replaced {dst_path}')
    else:
        print(f'Missing {src_path}')
