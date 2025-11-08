from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Tuple, Dict, Any
from math import hypot
from itertools import permutations
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Point(BaseModel):
    label: str
    x: float
    y: float

class SolveRequest(BaseModel):
    points: List[Point]
    algorithm: Literal['bruteforce','greedy','divide_and_conquer','dfs','bfs']
    start_index: int = 0


def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return hypot(a[0]-b[0], a[1]-b[1])

# Helpers to convert

def to_coords(points: List[Point]):
    return [(p.x,p.y) for p in points]

# 1) Bruteforce — check all permutations (warning: O(n!))

def bruteforce(points_coords, start=0):
    n = len(points_coords)
    indices = list(range(n))
    other = [i for i in indices if i != start]
    best = None
    best_len = float('inf')
    log = []
    for perm in permutations(other):
        route = [start] + list(perm) + [start]
        d = 0
        steps = []
        for i in range(len(route)-1):
            a, b = route[i], route[i+1]
            jarak = dist(points_coords[a], points_coords[b])
            d += jarak
            steps.append(f"Dari {a} ke {b}, jarak: {jarak:.2f}, total: {d:.2f}")
        log.append({
            'rute': [start] + list(perm) + [start],
            'jarak_total': d,
            'langkah': steps
        })
        if d < best_len:
            best_len = d
            best = route
    # Konversi log menjadi string agar lebih mudah dibaca di frontend
    readable_log = []
    for entry in log:
        readable_log.append(f"Rute: {entry['rute']}, Total jarak: {entry['jarak_total']:.2f}")
        readable_log.extend(entry['langkah'])
    return {'best_route': best, 'best_distance': best_len, 'log': readable_log}

# 2) Greedy — nearest neighbor

def greedy(points_coords, start=0):
    n = len(points_coords)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    d = 0
    log = []
    while unvisited:
        cur = route[-1]
        next_node = min(unvisited, key=lambda x: dist(points_coords[cur], points_coords[x]))
        jarak = dist(points_coords[cur], points_coords[next_node])
        d += jarak
        log.append(f"Dari {cur} ke {next_node}, jarak: {jarak:.2f}, total: {d:.2f}")
        route.append(next_node)
        unvisited.remove(next_node)
    jarak_kembali = dist(points_coords[route[-1]], points_coords[start])
    d += jarak_kembali
    log.append(f"Dari {route[-1]} kembali ke {start}, jarak: {jarak_kembali:.2f}, total: {d:.2f}")
    route.append(start)
    return {'best_route': route, 'best_distance': d, 'log': log}

# 3) Divide & Conquer (heuristic): split by median x or y, solve recursively, then merge with simple stitch + 2-opt limited passes

def two_opt(route, pts, max_iter=100):
    improved = True
    it = 0
    def route_len(r):
        s = 0
        for i in range(len(r)-1):
            s += dist(pts[r[i]], pts[r[i+1]])
        return s
    best = route[:]
    best_len = route_len(best)
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j - i == 1: continue
                newr = best[:i] + best[i:j][::-1] + best[j:]
                nl = route_len(newr)
                if nl < best_len:
                    best = newr
                    best_len = nl
                    improved = True
    return best, best_len


def divide_and_conquer(points_coords, start=0):
    def solve_recursive(indices):
        if len(indices) <= 2:
            route = indices + [indices[0]]
            distance = sum(dist(points_coords[route[i]], points_coords[route[i + 1]]) for i in range(len(route) - 1))
            return route, distance, [f"Langsung hitung jarak untuk {route}: {distance:.2f}"]
        indices.sort(key=lambda i: points_coords[i][0])
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]
        left_route, left_distance, left_log = solve_recursive(left_indices)
        right_route, right_distance, right_log = solve_recursive(right_indices)
        merged_route = left_route[:-1] + right_route[:-1] + [left_route[0]]
        merged_distance = sum(dist(points_coords[merged_route[i]], points_coords[merged_route[i + 1]]) for i in range(len(merged_route) - 1))
        merge_log = [f"Gabungkan rute kiri {left_route} dan kanan {right_route}, total jarak: {merged_distance:.2f}"]
        return merged_route, merged_distance, left_log + right_log + merge_log

    indices = list(range(len(points_coords)))
    indices.remove(start)
    route, distance, log = solve_recursive([start] + indices)
    return {'best_route': route, 'best_distance': distance, 'log': log}

# 4) DFS backtracking (like bruteforce but explicit stack and can return partial logs)

def dfs_backtrack(points_coords, start=0):
    n = len(points_coords)
    best = None
    best_len = float('inf')
    log = []
    visited = [False]*n
    path = [start]
    visited[start] = True
    def backtrack(current, cum, depth):
        nonlocal best, best_len
        if depth == n:
            total = cum + dist(points_coords[current], points_coords[start])
            route = path[:] + [start]
            log.append(f"Rute selesai: {route}, total jarak: {total:.2f}")
            if total < best_len:
                best_len = total
                best = route[:]
            return
        for nxt in range(n):
            if not visited[nxt]:
                visited[nxt] = True
                path.append(nxt)
                log.append(f"Jelajahi dari {current} ke {nxt}, jarak saat ini: {cum:.2f}")
                backtrack(nxt, cum + dist(points_coords[current], points_coords[nxt]), depth+1)
                path.pop()
                visited[nxt] = False
    backtrack(start, 0, 1)
    return {'best_route': best, 'best_distance': best_len, 'log': log}

# 5) BFS (level by level partial path expansion) — extremely heavy for n>9

def bfs_search(points_coords, start=0):
    from collections import deque
    n = len(points_coords)
    q = deque()
    q.append({'route': [start], 'visited': {start}, 'cum': 0})
    best = None
    best_len = float('inf')
    log = []
    while q:
        node = q.popleft()
        if len(node['route']) == n:
            total = node['cum'] + dist(points_coords[node['route'][-1]], points_coords[start])
            route = node['route'] + [start]
            log.append(f"Rute selesai: {route}, total jarak: {total:.2f}")
            if total < best_len:
                best_len = total
                best = route
            continue
        for nxt in range(n):
            if nxt not in node['visited']:
                newcum = node['cum'] + dist(points_coords[node['route'][-1]], points_coords[nxt])
                log.append(f"Tambahkan {nxt} ke rute {node['route']}, jarak saat ini: {newcum:.2f}")
                q.append({'route': node['route'] + [nxt], 'visited': set(node['visited']) | {nxt}, 'cum': newcum})
    return {'best_route': best, 'best_distance': best_len, 'log': log}


@app.post('/solve')
def solve(req: SolveRequest):
    pts = to_coords(req.points)
    alg = req.algorithm
    start = req.start_index if 0 <= req.start_index < len(pts) else 0
    if alg == 'bruteforce':
        res = bruteforce(pts, start)
    elif alg == 'greedy':
        res = greedy(pts, start)
    elif alg == 'divide_and_conquer':
        res = divide_and_conquer(pts, start)
    elif alg == 'dfs':
        res = dfs_backtrack(pts, start)
    elif alg == 'bfs':
        res = bfs_search(pts, start)
    else:
        return {'error': 'unknown algorithm'}
    # map indices back to labels and coords in a friendly response
    route = res['best_route']
    route_info = []
    if route:
        for idx in route:
            route_info.append({
                'index': idx,
                'label': req.points[idx].label,
                'x': req.points[idx].x,
                'y': req.points[idx].y
            })
    return {
        'algorithm': alg,
        'best_distance': res.get('best_distance'),
        'route': route_info,
        'raw_log': res.get('log')
    }