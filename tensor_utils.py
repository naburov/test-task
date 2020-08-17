import numpy as np

SMALL_ANCHOR_BOXES = [[0.14765803503427266, 0.22079683929931457], [0.1930190345368917, 0.12853218210361067],
                      [0.08882472826086957, 0.11159594481605349]]
MEDIUM_ANCHOR_BOXES = [[0.20850895679662804, 0.3175711275026344], [0.30404776674937967, 0.2162298387096774],
                       [0.5222355769230769, 0.16105769230769232]]
LARGE_ANCHOR_BOXES = [[0.3607068607068607, 0.3765592515592516], [0.685378959276018, 0.4104920814479638],
                      [0.3924278846153846, 0.6146834935897436]]


def create_offsets(n_grid):
    offsets = np.zeros(shape=(4, 3, n_grid, n_grid, 2))
    for i in range(0, 1):
        for j in range(0, 3):
            for k in range(0, n_grid):
                for m in range(0, n_grid):
                    offsets[i, j, k, m, 0] = k
                    offsets[i, j, k, m, 1] = m
    return offsets


def create_anchor_boxes_tensor(boxes, n_grid):
    tensor = np.zeros(shape=(4, len(boxes), n_grid, n_grid, 2))
    for i in range(0, 1):
        for j in range(0, len(boxes)):
            for k in range(0, n_grid):
                for m in range(0, n_grid):
                    tensor[i, j, k, m, 0] = boxes[j][0]
                    tensor[i, j, k, m, 1] = boxes[j][1]
    return tensor
