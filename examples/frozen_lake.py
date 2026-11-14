"""冰湖寻路：多轮任务示例，用来比较 GRPO 和 GiGPO。

地图是一个方格，起点在左上角，终点在右下角，中间有若干个洞。
模型每一步只回答一个方向，走到终点成功，掉进洞里失败。
同一道题的地图固定，所以同组的轨迹会反复经过相同的位置，
这正是 GiGPO 的步级分组需要的条件。
"""

import random
from collections import deque

DIRECTIONS = {
    "上": (-1, 0),
    "下": (1, 0),
    "左": (0, -1),
    "右": (0, 1),
}

HOLE = "H"
ICE = "."


def shortest_distances_to_goal(grid):
    """从终点反向做广度优先搜索，返回每个可走格子到终点的最短步数。"""
    size = len(grid)
    goal = (size - 1, size - 1)
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        row, column = queue.popleft()
        for row_change, column_change in DIRECTIONS.values():
            neighbor = (row + row_change, column + column_change)
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if grid[neighbor[0]][neighbor[1]] == HOLE or neighbor in distances:
                continue
            distances[neighbor] = distances[(row, column)] + 1
            queue.append(neighbor)
    return distances


def generate_grid(random_generator, grid_size, hole_count):
    """随机放洞，直到起点能走到终点为止。"""
    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)
    candidates = [
        (row, column)
        for row in range(grid_size)
        for column in range(grid_size)
        if (row, column) not in (start, goal)
    ]
    while True:
        holes = set(random_generator.sample(candidates, hole_count))
        grid = [
            "".join(HOLE if (row, column) in holes else ICE for column in range(grid_size))
            for row in range(grid_size)
        ]
        if start in shortest_distances_to_goal(grid):
            return grid


def choose_start(random_generator, grid, minimum_start_distance):
    """在离终点足够远、且有路可走的格子里随机选一个起点。"""
    distances = shortest_distances_to_goal(grid)
    candidates = sorted(
        cell for cell, distance in distances.items() if distance >= minimum_start_distance
    )
    if not candidates:
        return None
    return random_generator.choice(candidates)


def make_dataset(size, seed=0, grid_size=4, hole_count=3, random_start=False,
                 minimum_start_distance=3):
    random_generator = random.Random(seed)
    dataset = []
    for _ in range(size):
        grid = generate_grid(random_generator, grid_size, hole_count)
        sample = {"grid": grid, "start": [0, 0]}
        if random_start:
            start = choose_start(random_generator, grid, minimum_start_distance)
            while start is None:      # 这张地图上没有足够远的起点，换一张
                grid = generate_grid(random_generator, grid_size, hole_count)
                start = choose_start(random_generator, grid, minimum_start_distance)
            sample = {"grid": grid, "start": list(start)}
        dataset.append(sample)
    return dataset


def render_grid(grid, start=(0, 0)):
    size = len(grid)
    lines = []
    for row in range(size):
        cells = []
        for column in range(size):
            if (row, column) == tuple(start):
                cells.append("起")
            elif (row, column) == (size - 1, size - 1):
                cells.append("终")
            elif grid[row][column] == HOLE:
                cells.append("洞")
            else:
                cells.append("冰")
        lines.append(" ".join(cells))
    return "\n".join(lines)


class FrozenLakeEnvironment:
    """reward_mode:
    - "sparse"：只有走到终点得 1 分，其余全是 0。
    - "progress"：每靠近终点一步给一点分，未到终点时整局合计不超过 0.5；
      掉进洞里会把已经拿到的进度分全部扣回，整局合计为 0。
    hole_penalty：掉进洞里额外扣的分，整局合计变成 -hole_penalty。
    两种 reward_mode 下都生效。
    step_cost：没有结束的每一步扣的分。成功和掉洞时会结清，整局合计仍恰好是 1 和 -hole_penalty，
    所以它只影响"活着但没到终点"的轨迹：拖得越久扣得越多。
    """

    def __init__(self, reward_mode="sparse", describe_neighbors=False, hole_penalty=0.0,
                 step_cost=0.0):
        if hole_penalty < 0 or step_cost < 0:
            raise ValueError("hole_penalty 和 step_cost 不能为负数")
        self.step_cost = step_cost
        self.describe_neighbors = describe_neighbors
        self.hole_penalty = hole_penalty
        if reward_mode not in ("sparse", "progress"):
            raise ValueError(f"未知的 reward_mode：{reward_mode!r}")
        self.reward_mode = reward_mode

    def reset(self, sample, seed):
        self.grid = sample["grid"]
        self.size = len(self.grid)
        self.distances = shortest_distances_to_goal(self.grid)
        self.start = tuple(sample.get("start", (0, 0)))
        self.position = self.start
        self.initial_distance = self.distances[self.position]
        self.successful = False
        self.fell_into_hole = False
        self.step_count = 0
        self.invalid_action_count = 0
        self.wall_hit_count = 0
        self.progress_reward_total = 0.0
        instructions = (
            f"你在一个 {self.size}×{self.size} 的冰湖上，要从起点走到右下角的终点。\n"
            "地图（第 1 行在最上面，第 1 列在最左边）：\n"
            f"{render_grid(self.grid, self.start)}\n"
            f"终点在第 {self.size} 行第 {self.size} 列。\n"
            "上：行号减 1；下：行号加 1；左：列号减 1；右：列号加 1。\n"
            "掉进洞里就失败。每次只回答一个字：上、下、左、右，不要输出任何其他内容。\n"
            f"{self._describe_position()}"
        )
        return [{"role": "user", "content": instructions}]

    def _describe_position(self):
        description = f"你现在在第 {self.position[0] + 1} 行第 {self.position[1] + 1} 列。"
        if self.describe_neighbors:
            description += self._describe_neighbors()
        return description

    def _describe_neighbors(self):
        goal = (self.size - 1, self.size - 1)
        parts = []
        for name, (row_change, column_change) in DIRECTIONS.items():
            target = (self.position[0] + row_change, self.position[1] + column_change)
            if not (0 <= target[0] < self.size and 0 <= target[1] < self.size):
                kind = "湖岸"
            elif self.grid[target[0]][target[1]] == HOLE:
                kind = "洞"
            elif target == goal:
                kind = "终点"
            else:
                kind = "冰"
            parts.append(f"{name}边是{kind}")
        return "，".join(parts) + "。"

    def step(self, action_text):
        self.step_count += 1
        action = action_text.strip()
        if action not in DIRECTIONS:
            self.invalid_action_count += 1
            feedback = "回答无效。只能回答一个字：上、下、左、右。" + self._describe_position()
            return [{"role": "user", "content": feedback}], self._charge_step_cost(), False

        row_change, column_change = DIRECTIONS[action]
        target = (self.position[0] + row_change, self.position[1] + column_change)
        if not (0 <= target[0] < self.size and 0 <= target[1] < self.size):
            self.wall_hit_count += 1
            feedback = "那边是湖岸，走不过去。" + self._describe_position()
            return [{"role": "user", "content": feedback}], self._charge_step_cost(), False

        if self.grid[target[0]][target[1]] == HOLE:
            self.fell_into_hole = True
            self.position = target
            # 扣回已经拿到的进度分，再减去掉洞的惩罚：整局合计为 -hole_penalty。
            penalty = -self.progress_reward_total - self.hole_penalty
            self.progress_reward_total = 0.0
            return [], penalty, True

        distance_before = self.distances[self.position]
        self.position = target
        distance_after = self.distances[self.position]

        if distance_after == 0:
            self.successful = True
            # 成功时整局合计恰好为 1，不叠加进度分。
            return [], 1.0 - self.progress_reward_total, True

        reward = self._charge_step_cost()
        if self.reward_mode == "progress":
            progress = 0.5 * (distance_before - distance_after) / self.initial_distance
            self.progress_reward_total += progress
            reward += progress
        return [{"role": "user", "content": self._describe_position()}], reward, False

    def _charge_step_cost(self):
        """没有结束的每一步都扣一点分，让原地不动不再是零成本。记入累计，结束时一并结清。"""
        self.progress_reward_total -= self.step_cost
        return -self.step_cost

    def state_key(self):
        return f"{self.position[0]},{self.position[1]}"

    def is_successful(self):
        return self.successful

    def metrics(self):
        step_count = max(self.step_count, 1)
        remaining = 0 if self.successful else self.distances.get(self.position, self.initial_distance)
        result = {
            "invalid_action_fraction": self.invalid_action_count / step_count,
            "wall_hit_fraction": self.wall_hit_count / step_count,
            "hole_fraction": 1.0 if self.fell_into_hole else 0.0,
            "remaining_distance_fraction": remaining / self.initial_distance,
        }
        if self.successful:
            result["extra_step_count"] = float(self.step_count - self.initial_distance)
        return result
