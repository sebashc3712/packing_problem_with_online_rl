from typing import List, Tuple, Dict, Any
import pickle, random, statistics

def generate_ga_mixed_instances(
    input_paths: List[str],
    output_path: str,
    generations: int = 40,
    population_size: int = 1000,
    elite_frac: float = 0.10,
    mutation_reorder_p: float = 0.15,
    mutation_switch_p: float = 0.10,
    mutation_dim_p: float = 0.08,
    target_fill: float = 0.85,
    alpha_diversity: float = 0.12,
    keep_episode_lengths: bool = True,
    n_output_episodes: int = 3000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Load episodes from multiple .pt files, evolve them with a GA using:
      - reorder mutation (change place of a box),
      - switch-between-episodes mutation (swap a box across episodes),
      - dimension mutation (±1 clamped),
    and save a single big .pt containing top-scoring episodes.

    The output file is a plain pickled Python list, so
    torch.load(path, weights_only=True) will return the list on your machine.
    """
    def _load_all_pickles(path: str) -> list:
        objs = []
        with open(path, "rb") as f:
            while True:
                try:
                    objs.append(pickle.load(f))
                except EOFError:
                    break
        return objs

    def _looks_like_episodes(obj: Any) -> bool:
        if not isinstance(obj, list) or not obj:
            return False
        first = obj[0]
        if not isinstance(first, list) or not first:
            return False
        first_box = first[0]
        return isinstance(first_box, list) and len(first_box) == 3 and all(isinstance(x, int) for x in first_box)

    def load_episodes_from_pt(path: str):
        # direct
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if _looks_like_episodes(obj):
                return obj
        except Exception:
            pass
        # scan
        objs = _load_all_pickles(path)
        for o in objs:
            if _looks_like_episodes(o):
                return o
        raise ValueError(f"Could not find episodes inside: {path}")

    def save_episodes_to_pt(path: str, episodes):
        with open(path, "wb") as f:
            pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    def episode_volume(ep):
        return sum(x*y*z for x,y,z in ep)

    def episode_diversity(ep):
        if not ep: return 0.0
        return len(set(tuple(b) for b in ep))/len(ep)

    def fitness(ep, capacity=(10,10,10), target_fill=0.80, alpha_diversity=0.15):
        cap_vol = capacity[0]*capacity[1]*capacity[2]
        v = episode_volume(ep)
        fill = v/cap_vol if cap_vol > 0 else 0
        score_fill = 1.0 - abs(fill - target_fill)
        score_div = episode_diversity(ep)
        return score_fill + alpha_diversity*score_div

    def crossover(a, b):
        if not a and not b: return []
        if not a: return list(b)
        if not b: return list(a)
        target_len = random.choice([len(a), len(b)])
        child, i, j, toggle = [], 0, 0, True
        while len(child) < target_len and (i < len(a) or j < len(b)):
            if toggle and i < len(a):
                child.append(a[i]); i += 1
            elif (not toggle) and j < len(b):
                child.append(b[j]); j += 1
            toggle = not toggle
        while len(child) < target_len and i < len(a):
            child.append(a[i]); i += 1
        while len(child) < target_len and j < len(b):
            child.append(b[j]); j += 1
        return child

    def mutate_reorder(ep, p_swap=0.15):
        n = len(ep)
        if n < 2: return
        num_swaps = max(1, int(p_swap*n))
        for _ in range(num_swaps):
            i, j = random.randrange(n), random.randrange(n)
            ep[i], ep[j] = ep[j], ep[i]

    def mutate_switch_between_episodes(population, p_switch=0.10):
        num_pairs = max(1, int(p_switch*len(population)))
        for _ in range(num_pairs):
            a, b = random.sample(range(len(population)), 2)
            if not population[a] or not population[b]:
                continue
            i = random.randrange(len(population[a]))
            j = random.randrange(len(population[b]))
            population[a][i], population[b][j] = population[b][j], population[a][i]

    def mutate_dimensions(ep, dim_min, dim_max, p_dim=0.08):
        for k in range(len(ep)):
            if random.random() < p_dim:
                x,y,z = ep[k]
                idx = random.randrange(3)
                
                # Hard mutation: occasionally jump to large dims
                if random.random() < 0.1:
                    if idx == 0: x = random.randint(3, 7)
                    elif idx == 1: y = random.randint(3, 7)
                    else: z = random.randint(2, 5)
                else:
                    if idx == 0: x += random.choice([-1,1])
                    elif idx == 1: y += random.choice([-1,1])
                    else: z += random.choice([-1,1])
                
                x = max(dim_min[0], min(dim_max[0], x))
                y = max(dim_min[1], min(dim_max[1], y))
                z = max(dim_min[2], min(dim_max[2], z))
                ep[k] = [int(x),int(y),int(z)]

    def tournament_select(pop, k=3):
        sample = random.sample(pop, k=min(k, len(pop)))
        scored = sorted(((fitness(ep), ep) for ep in sample), key=lambda t: t[0], reverse=True)
        return scored[0][1]

    def build_initial_population(pools, pop_size, keep_episode_lengths=True):
        population = []
        all_boxes = [b for pool in pools for ep in pool for b in ep]
        for _ in range(pop_size):
            src = random.choice(pools)
            ep = random.choice(src)
            if keep_episode_lengths:
                population.append([b[:] for b in ep])
            else:
                target_len = max(1, int(random.gauss(len(ep), max(1,len(ep)//10))))
                new_ep = []
                start = random.randrange(0, max(1,len(ep)))
                new_ep.extend(ep[start:start+target_len])
                while len(new_ep) < target_len:
                    new_ep.append(random.choice(all_boxes))
                population.append([b[:] for b in new_ep[:target_len]])
        return population

    def evolve_population(pools, generations=40, population_size=1000, elite_frac=0.10,
                          mutation_reorder_p=0.15, mutation_switch_p=0.10, mutation_dim_p=0.05,
                          target_fill=0.80, alpha_diversity=0.15, keep_episode_lengths=True, random_seed=42):
        random.seed(random_seed)
        all_boxes = [b for pool in pools for ep in pool for b in ep]
        dim_min = tuple(min(d[i] for d in all_boxes) for i in range(3))
        dim_max = tuple(max(d[i] for d in all_boxes) for i in range(3))

        population = build_initial_population(pools, population_size, keep_episode_lengths=keep_episode_lengths)

        for _ in range(generations):
            scored = sorted(((fitness(ep, target_fill=target_fill, alpha_diversity=alpha_diversity), ep) for ep in population),
                            key=lambda t: t[0], reverse=True)
            elite_count = max(1, int(elite_frac * len(scored)))
            new_population = [scored[i][1] for i in range(elite_count)]
            while len(new_population) < population_size:
                p1 = tournament_select(population, k=3)
                p2 = tournament_select(population, k=3)
                child = crossover(p1, p2)
                mutate_reorder(child, p_swap=mutation_reorder_p)                 # (1) change place
                mutate_dimensions(child, dim_min=dim_min, dim_max=dim_max, p_dim=mutation_dim_p)  # (3) change dims
                new_population.append(child)
            mutate_switch_between_episodes(new_population, p_switch=mutation_switch_p)           # (2) switch between episodes
            population = new_population
        return population

    # 1) Load all source episodes
    pools = [load_episodes_from_pt(p) for p in input_paths]

    # 2) Evolve
    random.seed(random_seed)
    pop = evolve_population(
        pools,
        generations=generations,
        population_size=population_size,
        elite_frac=elite_frac,
        mutation_reorder_p=mutation_reorder_p,
        mutation_switch_p=mutation_switch_p,
        mutation_dim_p=mutation_dim_p,
        target_fill=target_fill,
        alpha_diversity=alpha_diversity,
        keep_episode_lengths=keep_episode_lengths,
        random_seed=random_seed
    )

    # 3) Select the top N episodes and save
    ranked = sorted(((fitness(ep, target_fill=target_fill, alpha_diversity=alpha_diversity), ep) for ep in pop),
                    key=lambda x: x[0], reverse=True)
    selected = [ep for _, ep in ranked[:n_output_episodes]] if n_output_episodes <= len(ranked) else [ep for _, ep in ranked]
    save_episodes_to_pt(output_path, selected)

    # 4) Return a small summary
    def summarize_episodes(episodes):
        lens = [len(ep) for ep in episodes]
        all_boxes = [b for ep in episodes for b in ep]
        mins = [min(dim) for dim in zip(*all_boxes)] if all_boxes else [None,None,None]
        maxs = [max(dim) for dim in zip(*all_boxes)] if all_boxes else [None,None,None]
        vols = [sum(b[0]*b[1]*b[2] for b in ep) for ep in episodes]
        return {
            "num_episodes": len(episodes),
            "min_boxes_per_episode": min(lens) if lens else 0,
            "max_boxes_per_episode": max(lens) if lens else 0,
            "avg_boxes_per_episode": (sum(lens)/len(lens)) if lens else 0,
            "min_dim": mins,
            "max_dim": maxs,
            "avg_episode_volume": (sum(vols)/len(vols)) if vols else 0,
        }

    src_summaries = []
    for p, pool in zip(input_paths, pools):
        src_summaries.append({"path": p, **summarize_episodes(pool)})

    return {
        "inputs": src_summaries,
        "output_path": output_path,
        "output_summary": summarize_episodes(selected),
        "ga_config": {
            "generations": generations,
            "population_size": population_size,
            "elite_frac": elite_frac,
            "mutation_reorder_p": mutation_reorder_p,
            "mutation_switch_p": mutation_switch_p,
            "mutation_dim_p": mutation_dim_p,
            "target_fill": target_fill,
            "alpha_diversity": alpha_diversity,
            "keep_episode_lengths": keep_episode_lengths,
            "n_output_episodes": n_output_episodes,
            "random_seed": random_seed
        }
    }


summary = generate_ga_mixed_instances(
    input_paths=["approachesO3DKP/cut_1.pt", "approachesO3DKP/cut_2.pt", "approachesO3DKP/rs.pt"],
    output_path="approachesO3DKP/ga_mixed.pt",
    generations=100,
    population_size=10000,
    n_output_episodes=10000,
    keep_episode_lengths=True,
    target_fill=0.85,
    alpha_diversity=0.12,
    random_seed=42
)
print(summary)
