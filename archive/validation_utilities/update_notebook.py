import json

# Read the notebook
with open('notebooks/experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the cell with tumor_contour = max(filtered_contours...)
updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'tumor_contour = max(filtered_contours' in source and not updated:
            # Replace with error handling
            new_source = '''if filtered_contours:
    tumor_contour = max(filtered_contours, key=cv2.contourArea)
else:
    print("Warning: No valid contours found. Check your filtering thresholds.")
    tumor_contour = None'''
            cell['source'] = [new_source]
            updated = True
            print('✓ Updated tumor_contour cell with error handling')
            break

if updated:
    # Write back
    with open('notebooks/experiment.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('✓ Notebook saved successfully')
else:
    print('✗ Cell with tumor_contour not found')
