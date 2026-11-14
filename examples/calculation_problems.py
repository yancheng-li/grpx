"""计算密集的应用题：推理只有两三步，但数字很大，心算几乎必错。

用来检验"模型能不能学会用计算器"：不用工具时准确率应该很低，用了应该很高。
题目由模板随机生成，答案都是整数，由程序精确算出。
样本的字段和 examples/gsm8k.py 相同（prompt、question、answer），
所以单轮任务和带计算器的多轮任务都可以直接使用，判定函数也沿用 gsm8k 的。
"""

import random

INSTRUCTION = ("Solve the problem step by step. "
               "End your response with a line of the form '#### <number>'.")

# "train" 和 "test" 用同一组模板、不同的随机数；
# "unseen" 用另一组训练时没见过的模板，用来检验学到的是不是只对这几种题型有效。
SPLIT_SEED_OFFSETS = {"train": 0, "test": 1_000_003, "unseen": 2_000_003}


def _warehouse(generator):
    boxes = generator.randint(1200, 9800)
    items = generator.randint(120, 980)
    damaged = generator.randint(1000, 90000)
    question = (f"A warehouse received {boxes} boxes, and each box holds {items} items. "
                f"During inspection, {damaged} items were found damaged and thrown away. "
                "How many items are left?")
    return question, boxes * items - damaged


def _payroll(generator):
    workers = generator.randint(150, 950)
    wage = generator.randint(610, 1890)
    weeks = generator.randint(12, 48)
    bonus = generator.randint(10000, 99000)
    question = (f"A factory employs {workers} workers and pays each of them {wage} dollars per week. "
                f"After {weeks} weeks, the factory also pays out {bonus} dollars in bonuses in total. "
                "How many dollars has the factory paid in wages and bonuses altogether?")
    return question, workers * wage * weeks + bonus


def _revenue(generator):
    first_units = generator.randint(1100, 8900)
    first_price = generator.randint(130, 970)
    second_units = generator.randint(1100, 8900)
    second_price = generator.randint(130, 970)
    question = (f"A store sold {first_units} phones at {first_price} dollars each in March "
                f"and {second_units} phones at {second_price} dollars each in April. "
                "What was the total revenue in dollars over the two months?")
    return question, first_units * first_price + second_units * second_price


def _equal_split(generator):
    teams = generator.randint(13, 97)
    share = generator.randint(21000, 98000)
    taxes = generator.randint(100000, 900000)
    total = share * teams + taxes
    question = (f"A company earned {total} dollars this year. It first set aside {taxes} dollars for taxes "
                f"and then divided the rest equally among its {teams} teams. "
                "How many dollars did each team receive?")
    return question, share


def _train(generator):
    speed = generator.randint(113, 289)
    hours = generator.randint(13, 23)
    days = generator.randint(120, 360)
    extra = generator.randint(1000, 9000)
    question = (f"A freight train travels at {speed} kilometers per hour for {hours} hours every day. "
                f"After {days} days it makes one extra trip of {extra} kilometers. "
                "How many kilometers has it traveled in total?")
    return question, speed * hours * days + extra


def _harvest_difference(generator):
    first_crates = generator.randint(1100, 8900)
    first_weight = generator.randint(110, 490)
    second_crates = generator.randint(1100, 8900)
    second_weight = generator.randint(110, 490)
    while first_crates * first_weight == second_crates * second_weight:
        second_crates += 1
    question = (f"One farm harvested {first_crates} crates of apples weighing {first_weight} kilograms each. "
                f"Another farm harvested {second_crates} crates weighing {second_weight} kilograms each. "
                "By how many kilograms does the larger harvest exceed the smaller one?")
    return question, abs(first_crates * first_weight - second_crates * second_weight)


def _inventory(generator):
    sold_per_day = generator.randint(310, 890)
    days = generator.randint(110, 280)
    deliveries = generator.randint(21, 89)
    per_delivery = generator.randint(1200, 4800)
    start = sold_per_day * days + generator.randint(5000, 60000)
    question = (f"A depot started the year with {start} bottles in stock. "
                f"It received {deliveries} deliveries of {per_delivery} bottles each, "
                f"and it sold {sold_per_day} bottles per day for {days} days. "
                "How many bottles are in stock now?")
    return question, start + deliveries * per_delivery - sold_per_day * days


def _tickets(generator):
    adults = generator.randint(2100, 9700)
    adult_price = generator.randint(115, 385)
    children = generator.randint(1100, 6900)
    child_price = generator.randint(45, 105)
    refunded = generator.randint(110, 990)
    question = (f"A concert sold {adults} adult tickets at {adult_price} dollars each "
                f"and {children} child tickets at {child_price} dollars each. "
                f"Later, {refunded} adult tickets were refunded at full price. "
                "How many dollars of ticket revenue remain?")
    return question, adults * adult_price + children * child_price - refunded * adult_price


def _field(generator):
    length = generator.randint(210, 940)
    width = generator.randint(130, 760)
    cost = generator.randint(13, 89)
    question = (f"A rectangular field is {length} meters long and {width} meters wide. "
                f"Laying turf costs {cost} dollars per square meter. "
                "How many dollars does it cost to cover the whole field?")
    return question, length * width * cost


def _profit(generator):
    chairs = generator.randint(2300, 9600)
    cost = generator.randint(117, 398)
    price = cost + generator.randint(105, 487)
    question = (f"A workshop built {chairs} chairs. Each chair cost {cost} dollars to make "
                f"and was sold for {price} dollars. "
                "How many dollars of profit did the workshop make in total?")
    return question, chairs * (price - cost)


def _savings(generator):
    rent = generator.randint(1150, 2950)
    food = generator.randint(410, 980)
    salary = rent + food + generator.randint(1200, 4700)
    months = generator.randint(37, 143)
    question = (f"Maria earns {salary} dollars every month. Each month she pays {rent} dollars for rent "
                f"and {food} dollars for food, and she saves everything that is left. "
                f"How many dollars has she saved after {months} months?")
    return question, (salary - rent - food) * months


def _printer(generator):
    pages = generator.randint(37, 96)
    hours = generator.randint(7, 19)
    days = generator.randint(120, 340)
    discarded = generator.randint(1100, 9800)
    question = (f"A printing machine prints {pages} pages per minute. It runs {hours} hours a day for {days} days. "
                f"Afterwards, {discarded} misprinted pages are thrown away. "
                "How many good pages are left?")
    return question, pages * 60 * hours * days - discarded


def _bottling(generator):
    bottle = generator.randint(13, 47)
    count = generator.randint(21000, 87000)
    spilled = generator.randint(1100, 9900)
    total = bottle * count + spilled
    question = (f"A dairy had {total} liters of milk. {spilled} liters were spilled, "
                f"and the rest was poured into containers that hold {bottle} liters each, filling every container completely. "
                "How many containers were filled?")
    return question, count


TEMPLATES = [_warehouse, _payroll, _revenue, _equal_split, _train,
             _harvest_difference, _inventory, _tickets]
UNSEEN_TEMPLATES = [_field, _profit, _savings, _printer, _bottling]


def make_dataset(split="train", size=2000, seed=0):
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"未知的 split：{split!r}，可选 {sorted(SPLIT_SEED_OFFSETS)}")
    generator = random.Random(seed + SPLIT_SEED_OFFSETS[split])
    templates = UNSEEN_TEMPLATES if split == "unseen" else TEMPLATES
    samples = []
    for index in range(size):
        question, answer = templates[index % len(templates)](generator)
        samples.append({
            "prompt": f"{question}\n\n{INSTRUCTION}",
            "question": question,
            "answer": float(answer),
        })
    generator.shuffle(samples)
    return samples
