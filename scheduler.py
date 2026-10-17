"""scheduler.py -- Phase 3 Operational Core (Scheduler & Routing).

The Scheduler is the operational heart of the agent. Each turn it:
    1. Scans the board state and builds a list of prioritized ``Task`` objects
       grouped into the ten urgency bands (Section 2.A of the plan).
    2. Pre-empts any worker whose load would threaten the shed and sends them to
       drop (Section 2.B / the DROP pre-emption).
    3. Matches the remaining workers to tasks band-by-band using greedy Manhattan
       distance and emits the exact engine action for each worker.

Provenance: every band priority is read from ``rules_validated.json``
(``daily_routines``) rather than hard-coded, so tuning happens in data only.

Coordinate convention (documented and self-consistent):
    A position is an ``(x, y)`` tuple, 0-indexed on the 10x10 board.
        x = column  -> EAST increases x, WEST decreases x
        y = row     -> SOUTH increases y, NORTH decreases y
    The NW quadrant is the top-left (small x, small y), matching the land-deed
    layout in the handover. The shed's four centre tiles are (4,4), (5,4),
    (4,5), (5,5); DROP/PICKUP are the only actions that work while merely
    standing on one of those tiles (every other action is "boots in the dirt").
    LOCKED tiles are walkable, so routing needs no obstacle avoidance.
"""

from market_model import MarketModel

# The four centre tiles that give access to the shed (DROP / PICKUP).
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))


class TurnLedger:
    """Single mutable projection of the shed across one turn's planning stages.

    Fixes the stale-state divergence (#4): ``main.py`` builds one ledger per
    turn and threads it through the market-order stage *and* the worker-
    assignment stage, so every planner reads the same post-order shed
    projection.  A fertilizer/animal BUY placed in the morning raises the
    projection that the later DROP pre-emption sees, instead of each stage
    recomputing from an independent observation snapshot that predates the
    orders actually emitted.

    It also carries the turn's *named* reservations (#10) so unrelated concerns
    stop sharing one anonymous integer:
        ``shed_projected`` -- items expected to occupy the shed after this
                              turn's buys (physical stock + incoming purchases).
        ``feed_reserved``  -- shed stock held back from selling to feed placed
                              animals (#6); an item -> qty map.
    Sells are intentionally NOT credited back: the physical goods still occupy
    the shed until the engine settles the order, so keeping them counted keeps
    the DROP/overflow logic conservative against a spill.
    """

    __slots__ = ("shed_projected", "feed_reserved")

    def __init__(self, shed_usage=0, feed_reserved=None):
        self.shed_projected = shed_usage
        self.feed_reserved = dict(feed_reserved) if feed_reserved else {}

    def commit_buy(self, qty=1):
        """Record ``qty`` bought units arriving into the shed this turn."""
        if qty and qty > 0:
            self.shed_projected += qty
        return self.shed_projected

    def reserve_feed(self, item, qty):
        """Hold ``qty`` of ``item`` back from selling (animal feed, #6/#10)."""
        if qty and qty > 0:
            self.feed_reserved[item] = self.feed_reserved.get(item, 0) + qty

    @property
    def usage(self):
        """The projected shed occupancy the assignment stage should plan on."""
        return self.shed_projected


def manhattan(a, b):
    """Manhattan (taxicab) distance between two ``(x, y)`` positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(pos, target):
    """Return a single one-step move action list toward ``target``.

    Movement is relative and one step per turn (engine contract). We close the
    x-axis first, then the y-axis; the exact order is arbitrary on an
    obstacle-free board as long as each step strictly reduces the distance.
    Returns ``[]`` when the worker already stands on the target tile.
    """
    px, py = pos
    tx, ty = target
    if px != tx:
        return ["EAST"] if tx > px else ["WEST"]
    if py != ty:
        return ["SOUTH"] if ty > py else ["NORTH"]
    return []


def nearest_shed_tile(pos):
    """Return the closest shed centre tile to ``pos`` (deterministic on ties)."""
    return min(SHED_TILES, key=lambda tile: manhattan(pos, tile))


def turn_phase(step, rules):
    """Single source of temporal truth for a turn (#20).

    Consolidates the ``step``-arithmetic gates that previously lived inline and
    duplicated across ``main.py`` and the scheduler (hour-of-day, day index,
    season length in days, and the endgame window).  Keeping them in one
    documented helper means the day/dawn/endgame boundaries are defined once and
    cannot drift between the live agent and the offline checks.

    Returns a plain dict:
        hour:         step within the current day (0 == dawn roll-over)
        day:          0-indexed day number
        hours_per_day / season_length / season_days
        in_endgame:   step has reached ``policy.endgame_start_turn``
        is_dawn:      hour == 0
        is_last_hour: hour == hours_per_day - 1

    (Per-crop *watering windows* remain data in ``crop_params`` -- they are not
    step-derived, so they stay where the crop rules live rather than here.)
    """
    constants = rules.get("constants", {}) if rules else {}
    policy = rules.get("policy", {}) if rules else {}
    hours_per_day = constants.get("hours_per_day", 24)
    season_length = constants.get("season_length", 720)
    endgame_start = policy.get("endgame_start_turn", 670)
    hour = step % hours_per_day
    return {
        "hour": hour,
        "day": step // hours_per_day,
        "hours_per_day": hours_per_day,
        "season_length": season_length,
        "season_days": season_length // hours_per_day,
        "in_endgame": step >= endgame_start,
        "is_dawn": hour == 0,
        "is_last_hour": hour == hours_per_day - 1,
    }


def _hungarian(costs):
    """Solve the minimum-cost assignment problem (Hungarian algorithm).

    ``costs`` is a rectangular matrix (list of lists) where ``costs[i][j]`` is
    the cost of assigning worker *i* to task *j*.  Returns a list of
    ``(worker_index, task_index)`` pairs for the assignment that minimises
    total cost.  Unmatched workers or tasks (when the matrix is non-square)
    are simply omitted from the result.

    Pure-Python O(n^3) implementation with no external dependencies, suitable
    for the small matrices the scheduler produces (≤11 workers × ~20 tasks).
    """
    if not costs or not costs[0]:
        return []

    nrows = len(costs)
    ncols = len(costs[0])
    n = max(nrows, ncols)

    # Pad to a square matrix with a large sentinel cost so extra rows/cols
    # are never the minimum-cost choice.  The sentinel must be larger than
    # any real cost but not so huge that it dominates the potential
    # arithmetic (which can cause the algorithm to miss the optimal
    # assignment on rectangular matrices).
    max_real = max(costs[i][j] for i in range(nrows) for j in range(ncols))
    BIG = max_real * n + 1
    matrix = [[costs[i][j] if i < nrows and j < ncols else BIG
               for j in range(n)] for i in range(n)]

    # --- Kuhn-Munkres (Hungarian) on the square matrix ---
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = BIG + 1  # +1 so the first minv[j] == BIG passes the < test
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # Extract the assignment: p[j] = i means task j is assigned to worker i.
    result = []
    for j in range(1, n + 1):
        i = p[j]
        if i > 0 and i <= nrows and j <= ncols:
            result.append((i - 1, j - 1))
    return result


class Task:
    """A single unit of work the scheduler wants done at a board tile.

    Attributes:
        kind:     one of WATER / HARVEST / PLANT / DIG / FEED / CARE / DROP, or
                  an animal-setup verb (BUILD_COOP / BUILD_PASTURE / PLACE).
        pos:      the ``(x, y)`` tile the worker must stand on to act.
        priority: the urgency band value (higher runs first).
        value:    in-band tie-breaker; used for HARVEST as the "+ crop value"
                  term so richer crops are picked first without crossing bands.
        crop:     crop type -- the argument for a PLANT action, or context.
        same_day: True for the watering task paired with a fresh PLANT (a seed
                  starts one miss from death and must be watered the same day).
        item:     argument for actions that take one (PICKUP <item>); also the
                  animal type carried for a PLACE.
        fetch:    an item that must be picked up from the shed *before* the
                  worker can perform ``kind`` at ``pos`` (FEED needs "WHEAT", a
                  PLACE needs the animal). None means "act directly on arrival".
    """

    __slots__ = ("kind", "pos", "priority", "value", "crop", "same_day", "item", "fetch")

    def __init__(self, kind, pos, priority, value=0.0, crop=None, same_day=False,
                 item=None, fetch=None):
        self.kind = kind
        self.pos = tuple(pos)
        self.priority = priority
        self.value = value
        self.crop = crop
        self.same_day = same_day
        self.item = item
        self.fetch = fetch

    def action(self):
        """The engine action list to emit when a worker stands on ``pos``."""
        if self.kind == "PLANT":
            return ["PLANT", self.crop]
        if self.kind == "PICKUP":
            return ["PICKUP", self.item]
        return [self.kind]

    def sort_key(self):
        """Descending sort key: band first, then in-band crop value."""
        return (self.priority, self.value)

    def __repr__(self):
        return (
            f"Task({self.kind}, pos={self.pos}, pri={self.priority}, "
            f"val={self.value}, crop={self.crop}, same_day={self.same_day}, "
            f"item={self.item}, fetch={self.fetch})"
        )

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.pos == other.pos
            and self.priority == other.priority
            and self.value == other.value
            and self.crop == other.crop
            and self.same_day == other.same_day
            and self.item == other.item
            and self.fetch == other.fetch
        )


class Scheduler:
    """Builds prioritized tasks and routes workers via greedy Manhattan matching."""

    def __init__(self):
        # Drop mode governs the DROP pre-emption threshold. main.py sets it each
        # turn: "normal" (80%), "overflow" (half shed, e.g. end of day), or
        # "endgame" (0 -- liquidate every load).
        self.drop_mode = "normal"
        # Per-turn memo for _choose_plant_crop (#12): a single (key, result)
        # pair keyed on (step, market-stock signature).  A new turn's key misses
        # and recomputes, so the cached value can never go stale.
        self._plant_choice_cache = None

    # ------------------------------------------------------------------ #
    # Task generation
    # ------------------------------------------------------------------ #
    def generate_tasks(self, obs, rules, care_capacity, market_stocks=None,
                       num_workers=None):
        """Scan the board and return a list of prioritized ``Task`` objects.

        ``obs`` is a plain dict describing the board this turn. Recognised keys
        (all optional -- absent means "nothing of that kind"):
            step:          current turn index (for the planting cash test).
            crops:         list of crop dicts, each with at least ``pos``. Other
                           fields: ``type``, ``misses`` (missed waterings),
                           ``ready`` (ripe to harvest), ``needs_water``,
                           ``in_bonus_window``, ``repeater``.
            weeds:         list of ``(x, y)`` tiles to dig out.
            empty_tiles:   list of plantable ``(x, y)`` tiles.
            market_stocks: dict of item -> current market inventory.

        ``num_workers`` (optional) is the size of the crew this turn.  When
        given, it bounds new planting by crew throughput (#21): a crop must be
        watered roughly daily, so the sustainable acreage is capped at
        ``num_workers * care_policy.max_crops_per_worker``.  When None (the
        offline checks and unit tests that only exercise the CareMonitor bound)
        no crew cap is applied, preserving the prior behaviour.
        """
        dr = rules.get("daily_routines", {})
        policy = rules.get("policy", {})
        animals_on = policy.get("ANIMALS_ENABLED", False)
        to_die = rules.get("constants", {}).get("missed_waterings_to_die", 2)
        step = obs.get("step", 0)

        crops = obs.get("crops") or []
        weeds = obs.get("weeds") or []
        empty_tiles = obs.get("empty_tiles") or []
        if market_stocks is None:
            market_stocks = obs.get("market_stocks") or {}

        tasks = []

        # --- Standing crops: harvest when ripe, water by urgency band --------
        for crop in crops:
            pos = crop.get("pos")
            if pos is None:
                continue
            pos = tuple(pos)
            ctype = crop.get("type")
            is_repeater = self._is_repeater(crop, rules, ctype)

            if crop.get("ready", False):
                # Band 9000; the crop's market value is an in-band tie-break so
                # a ripe melon is picked before a ripe wheat but never outranks
                # the animal-harvest band above it.
                tasks.append(
                    Task(
                        "HARVEST",
                        pos,
                        dr["HARVEST_CROP"]["priority"],
                        value=self._crop_value(rules, ctype, market_stocks),
                        crop=ctype,
                    )
                )
                # Per-tile dedupe (#11): a ripe crop is harvested this turn, not
                # watered. Emitting a WATER for the same tile would compete for a
                # worker and, if won, waste the action on a crop that is about to
                # be removed by the harvest. One tile -> one crop task per turn.
                continue

            water_priority = self._water_priority(crop, dr["WATER"], to_die, is_repeater)
            if water_priority is not None:
                tasks.append(Task("WATER", pos, water_priority, crop=ctype))

        # --- Weeds: dig them out --------------------------------------------
        for weed in weeds:
            tasks.append(Task("DIG", tuple(weed), dr["DIG"]["priority"]))

        # --- Animal lifecycle tasks (computed up front for the #13 gate) -----
        # Built here (not appended last) so the planting stage below can see
        # whether any animal home still needs BUILDing this turn.  When animals
        # are disabled this is an empty list and nothing downstream changes.
        animal_tasks = self._animal_tasks(obs, rules) if animals_on else []
        infra_pending = any(t.kind.startswith("BUILD_") for t in animal_tasks)

        # --- Planting (within capacity + cash test) + same-day watering ------
        # PLANT-vs-infrastructure arbitration (#13): while an animal home is
        # still unbuilt, suppress speculative new planting so the crew commits
        # to raising the COOP/PASTURE first instead of splitting effort between
        # breaking ground and starting crops that will then compete for water.
        # Gated behind ANIMALS_ENABLED via ``infra_pending`` -- with animals off
        # this is always False and planting is unchanged.
        plant_crop = self._choose_plant_crop(rules, step, market_stocks)
        if plant_crop is not None and not infra_pending:
            effective_cap = care_capacity
            if num_workers is not None:
                # Crew-throughput bound (#21): a standing crop needs servicing
                # (watering) roughly daily, so the number a crew can sustain is
                # capped by ``care_policy.max_crops_per_worker`` per worker.  We
                # take the tighter of that ceiling and the CareMonitor's
                # (slower-moving) acreage target so we never plant more than the
                # crew can actually keep alive.
                per_worker = rules.get("care_policy", {}).get("max_crops_per_worker", 8)
                effective_cap = min(care_capacity, num_workers * per_worker)
            free_slots = max(0, effective_cap - len(crops))
            for tile in empty_tiles[:free_slots]:
                pos = tuple(tile)
                tasks.append(Task("PLANT", pos, dr["PLANT"]["priority"], crop=plant_crop))
                # A newly planted seed already carries one missed-water mark, so
                # it is effectively "about to die": pair a dying-priority water
                # task for the same tile. assign_tasks defers it until the plant
                # actually lands (see below) so no one waters bare ground.
                tasks.append(
                    Task(
                        "WATER",
                        pos,
                        dr["WATER"]["priority_dying"],
                        crop=plant_crop,
                        same_day=True,
                    )
                )
                # Note: same-day water tasks use priority_dying (9500) so they
                # outrank most other work, but assign_tasks re-bands them just
                # below PLANT (6000) so a different worker is assigned to water
                # *after* the plant worker has been dispatched.

        # --- Animal chores are skipped entirely while animals are disabled ---
        # (Prevention, not post-hoc filtering -- see design doc guardrails.)
        # When enabled, _animal_tasks (computed above) walks each animal's
        # lifecycle stage and emits HARVEST_ANIMAL (9100), FEED (8800),
        # CARE (4000), plus the setup BUILD_* / PLACE work.
        tasks.extend(animal_tasks)

        return tasks

    def _animal_tasks(self, obs, rules):
        """Build the animal-lifecycle tasks for this turn.

        Driven entirely by ``obs["animals"]`` -- one dict per animal slot the
        agent operates, mirroring how crops/weeds/empty_tiles drive crop work.
        Recognised per-animal fields (all optional, sensible defaults):
            type:        animal key into ``animal_params`` (GOOSE/COW/SHEEP).
            home_pos:    the ``(x, y)`` tile of its home (build + act target).
            home_built:  the COOP/PASTURE structure exists.
            owned:       we have bought it (it is in the shed or already placed).
            placed:      it is standing in its home and producing.
            fed_today:   it has been fed this day (miss 2 days -> it escapes).
            ready:       a product (egg/milk/wool) is ready to HARVEST.
            needs_care:  a CARE productivity bonus is available to bank.

        Each animal contributes only the task(s) matching its current stage of
        Build home -> Buy -> Pickup+Place -> daily Feed + Care -> Harvest.
        Buying is a market order (see generate_market_orders); the rest is
        worker work routed through the normal band matching.
        """
        dr = rules.get("daily_routines", {})
        animal_params = rules.get("animal_params", {})
        animals = obs.get("animals") or []
        market_stocks = obs.get("market_stocks") or {}
        shed = obs.get("shed") or {}
        tasks = []

        # #14: per-turn WHEAT pickup reservation for FEED.  Each FEED fetches
        # one wheat from the shed; when the shed holds SOME wheat but fewer
        # units than the unfed placed flock needs, ration the known supply so we
        # never dispatch more simultaneous wheat-pickups than there is wheat to
        # pick up (empty-handed PICKUPs waste worker-turns and show up as wasted
        # actions in the audit).  A reported-empty shed (0 wheat) is treated as
        # "supply unknown / incoming" and does NOT suppress the need-driven FEED
        # task -- feeding stays a need the routing satisfies once wheat arrives.
        feed_per_animal = rules.get("animal_policy", {}).get("feed_per_animal", 1)
        wheat_available = int(shed.get("WHEAT", 0) or 0)
        unfed = sum(1 for a in animals
                    if a.get("placed", False) and not a.get("fed_today", False)
                    and animal_params.get(a.get("type")) and a.get("home_pos") is not None)
        feed_budget = None
        if wheat_available > 0:
            feedable = wheat_available // max(1, feed_per_animal)
            if feedable < unfed:
                feed_budget = feedable

        for animal in animals:
            atype = animal.get("type")
            params = animal_params.get(atype)
            hp = animal.get("home_pos")
            if not params or hp is None:
                continue
            home = tuple(hp)

            # Stage 1: build the home (COOP for geese, PASTURE for cows/sheep).
            if not animal.get("home_built", False):
                build_kind = self._build_action(params.get("home"))
                if build_kind is not None:
                    tasks.append(Task(build_kind, home, dr["BUILD_HOME"]["priority"], crop=atype))
                continue

            # Stage 2: transport a bought-but-unplaced animal. The worker fetches
            # it from the shed (PICKUP), carries it, and PLACEs it on the home.
            if animal.get("owned", False) and not animal.get("placed", False):
                tasks.append(
                    Task("PLACE", home, dr["PLACE_ANIMAL"]["priority"],
                         crop=atype, item=atype, fetch=atype)
                )
                continue

            if not animal.get("placed", False):
                continue

            # Stage 3: a placed animal earns its keep. Bands order these:
            # harvest (9100) > feed (8800) > care (4000).
            if animal.get("ready", False):
                product = params.get("produces")
                tasks.append(
                    Task(
                        "HARVEST",
                        home,
                        dr["HARVEST_ANIMAL"]["priority"],
                        value=self._crop_value(rules, product, market_stocks),
                        crop=atype,
                    )
                )
            if not animal.get("fed_today", False):
                # Feeding needs one wheat carried from the shed first.  Respect
                # the per-turn wheat ration (#14): emit until the known supply
                # is spent; feed_budget is None when supply is ample or unknown.
                if feed_budget is None or feed_budget > 0:
                    tasks.append(Task("FEED", home, dr["FEED"]["priority"], crop=atype, fetch="WHEAT"))
                    if feed_budget is not None:
                        feed_budget -= 1
            if animal.get("needs_care", False):
                tasks.append(Task("CARE", home, dr["CARE"]["priority"], crop=atype))

        return tasks

    # ------------------------------------------------------------------ #
    # Market order generation (purchases)
    # ------------------------------------------------------------------ #
    def generate_market_orders(self, obs, rules, shed_usage=0, ledger=None):
        """Return the ``BUY_PRODUCT`` market orders to place this turn.

        These are engine ``market`` orders (plain lists), emitted alongside the
        morning HIRE/SELL routine in ``main.py``. Purchasing is disabled during
        endgame liquidation (the caller simply does not call us then), so the
        only concern here is the 100-item shed capacity: bought goods land in
        the shed, so we never order more than would fit.

        The shed projection is tracked in a shared ``TurnLedger`` (#4): when the
        caller threads one in, each buy we commit raises the projection that the
        later worker-assignment DROP stage also reads, so both stages plan from
        the same post-order state.  When no ledger is supplied we build a local
        one seeded from ``shed_usage`` (backward-compatible with the audit and
        unit-test call sites), which behaves exactly as the old running integer.

        ``BUY_PRODUCT`` is a single opcode; the item (``"FERTILIZER"``, or an
        animal type in Phase 4 step 2) is an argument, never a separate op.
        """
        policy = rules.get("policy", {})
        if ledger is None:
            ledger = TurnLedger(shed_usage)
        orders = []

        # --- Fertilizer -----------------------------------------------------
        if policy.get("FERTILIZER_ENABLED", False):
            qty = self._fertilizer_buy_qty(obs, rules, ledger.shed_projected)
            if qty > 0:
                orders.append(["BUY_PRODUCT", "FERTILIZER", qty])
                ledger.commit_buy(qty)

        # --- Animals: buy an animal once its home is built and it is not yet
        # owned. A bought animal arrives in the shed, so it takes one shed slot
        # until a worker carries it out to its home -- respect capacity.
        if policy.get("ANIMALS_ENABLED", False):
            shed_size = rules.get("constants", {}).get("shed_size", 100)
            animal_params = rules.get("animal_params", {})
            animals = obs.get("animals") or []
            shed = obs.get("shed") or {}
            feed_per_animal = rules.get("animal_policy", {}).get("feed_per_animal", 1)

            # Feed-coverage bootstrap gate (#3): only expand the flock when the
            # wheat on hand already covers a day's feed for every *placed*
            # animal.  Scoping the requirement to PLACED animals is deliberate:
            # an unplaced animal eats nothing yet, so the very first animal can
            # always be bought to bootstrap the economy even with an empty shed
            # (the feed gets bought/grown before it is placed).  Once animals
            # are producing, we stop buying more than we can feed.
            placed_count = sum(1 for a in animals if a.get("placed", False))
            wheat_on_hand = shed.get("WHEAT", 0) or 0
            feed_covered = wheat_on_hand >= feed_per_animal * placed_count

            # Per-home occupancy for the capacity gate (#7): how many animals are
            # already committed (owned or placed) to each home tile, so a buy
            # never over-fills a COOP/PASTURE beyond ``_home_capacity``.
            committed_per_home = {}
            for a in animals:
                if a.get("owned", False) or a.get("placed", False):
                    hp = a.get("home_pos")
                    if hp is not None:
                        key = tuple(hp)
                        committed_per_home[key] = committed_per_home.get(key, 0) + 1

            for animal in animals:
                atype = animal.get("type")
                params = animal_params.get(atype)
                if params is None:
                    continue
                if not animal.get("home_built", False) or animal.get("owned", False):
                    continue
                if not feed_covered:
                    continue  # #3: don't buy what the placed flock can't be fed alongside
                hp = animal.get("home_pos")
                home_cap = self._home_capacity(rules, params.get("home"))
                key = tuple(hp) if hp is not None else None
                if key is not None and committed_per_home.get(key, 0) >= home_cap:
                    continue  # #7: the home is already full
                if ledger.shed_projected + 1 <= shed_size:
                    orders.append(["BUY_PRODUCT", atype])
                    ledger.commit_buy(1)
                    if key is not None:
                        committed_per_home[key] = committed_per_home.get(key, 0) + 1

        return orders

    @staticmethod
    def _home_capacity(rules, home_type):
        """Maximum animals a home of ``home_type`` can hold.

        Single source of truth (#7) read by both the buy limit above and any
        PLACE gating: reads ``animal_policy.home_capacity`` (COOP/PASTURE) and
        falls back to ``default_home_capacity`` so an unrecognised home type
        still yields a finite, sane cap rather than unbounded placement.
        """
        animal_policy = rules.get("animal_policy", {})
        caps = animal_policy.get("home_capacity", {})
        default = animal_policy.get("default_home_capacity", 4)
        return caps.get(home_type, default)

    @staticmethod
    def _sell_orders(shed, rules=None, endgame=False, market_stocks=None, reserve=None):
        """A SELL order for shed items that should be sold this turn.

        Only items with a ``market_params`` entry are considered -- inputs like
        FERTILIZER or an unbought animal type are not sellable products.

        During normal play, premium goods (normal_price >=
        ``market_heuristics.premium_price_threshold``) are held when their
        current market price has collapsed below
        ``market_heuristics.depressed_price_fraction`` of normal, because
        selling into a crashed market destroys optionality.  Basic goods are
        always sold -- their price curves are gentle enough that holding
        provides no meaningful upside, and they clog the shed.  Both dials are
        read from the rule table (#6) rather than hard-coded so they are tuned
        in data alone.

        ``reserve`` (item -> qty) is stock held back from selling this turn --
        e.g. WHEAT kept to feed the placed flock (#6).  Reserved units are
        subtracted from the sellable quantity before an order is built.

        Orders are returned ranked by projected sale revenue, highest first
        (#5): under the engine's 10-order cap the most valuable liquidations are
        then the ones guaranteed to survive truncation.  Ties break on item
        name for determinism.

        During endgame, everything is liquidated regardless of price -- there
        is no future to hold for.
        """
        if not shed:
            return []
        market = rules.get("market_params", {}) if rules else {}
        heuristics = rules.get("market_heuristics", {}) if rules else {}
        premium_threshold = heuristics.get("premium_price_threshold", 100)
        depressed_fraction = heuristics.get("depressed_price_fraction", 0.5)
        orders = []
        for item, qty in shed.items():
            if not qty or int(qty) <= 0:
                continue
            if market is not None and item not in market:
                continue
            qty = int(qty)
            # Hold back any reserved units (animal feed, #6) before selling.
            if reserve:
                qty -= int(reserve.get(item, 0) or 0)
                if qty <= 0:
                    continue
            if not endgame and market is not None:
                params = market.get(item, {})
                normal_price = params.get("normal_price", 0)
                # For premium goods, check whether the current price is
                # depressed.  When market_stocks is unavailable (the audit
                # path) we approximate current stock as 0 -- a conservative
                # check: if even at zero stock the price is already below the
                # depressed fraction, the market is truly crashed and we hold.
                if normal_price >= premium_threshold:
                    stock = market_stocks.get(item, 0) if market_stocks else 0
                    live_price = MarketModel.market_price(rules, item, stock)
                    if live_price < normal_price * depressed_fraction:
                        continue  # hold -- price is depressed
            orders.append(["SELL", item, qty])
        # Revenue-rank (#5): sell the most valuable inventory first so the
        # 10-order engine cap never drops a high-value liquidation in favour of
        # a cheap one.  Deterministic tie-break on item name.
        if rules and len(orders) > 1:
            def _revenue(order):
                _op, oitem, oqty = order
                stock = market_stocks.get(oitem, 0) if market_stocks else 0
                return MarketModel.avg_price(rules, oitem, stock, oqty) * oqty
            orders.sort(key=lambda o: (-_revenue(o), o[1]))
        return orders

    def daily_market_orders(self, state, rules, hour, in_endgame, shed_usage=0, ledger=None):
        """The full ordered market plan for one turn.

        Encodes the handover's morning-then-overlay ladder in one place so both
        the live agent (``main.py``) and the offline compliance audit emit an
        identical bucket (they previously duplicated this and could drift):
            * hour 0        -> SELL the shed inventory first (so harvest cash is
                              banked), then HIRE the crew up to
                              ``target_hands`` with whatever market slots
                              remain.  Hiring is skipped entirely during
                              endgame -- late-season hands cannot produce enough
                              to justify their wage.
            * not endgame   -> input purchases (fertilizer / animals) from
                               ``generate_market_orders``.
            * endgame, h!=0 -> liquidate the shed every remaining turn.
        Orders are returned uncapped; the ``ActionEmitter`` applies the 10-order
        engine cap at emission.  Placing SELL before HIRE ensures that the
        10-order cap never starves the morning sell when ``target_hands`` is 10.

        A ``TurnLedger`` may be threaded in (#4) so the buys emitted here raise
        the shed projection the caller later hands to ``assign_tasks``; when
        absent one is built locally from ``shed_usage`` (unchanged behaviour for
        the audit / unit-test call sites).
        """
        policy = rules.get("policy", {}) if rules else {}
        shed = state.get("shed") or {}
        market_stocks = state.get("market_stocks") or {}
        if ledger is None:
            ledger = TurnLedger(shed_usage)
        max_orders = rules.get("constants", {}).get("max_market_orders", 10) if rules else 10

        # Reserve WHEAT to feed the placed flock (#6/#10) so feed stock is not
        # sold out from under animals that still need it.  Gated + endgame-off:
        # with animals disabled there are no placed animals (nothing reserved),
        # and endgame liquidates everything regardless, so selling is unchanged.
        if policy.get("ANIMALS_ENABLED", False) and not in_endgame:
            animals = state.get("animals") or []
            placed = sum(1 for a in animals if a.get("placed", False))
            feed_per_animal = rules.get("animal_policy", {}).get("feed_per_animal", 1)
            ledger.reserve_feed("WHEAT", placed * feed_per_animal)

        orders = []
        if hour == 0:
            # Sell first -- the harvest must be banked before market slots are
            # spent on hiring.  Hiring fills whatever slots remain.
            sell = self._sell_orders(shed, endgame=in_endgame, rules=rules,
                                     market_stocks=market_stocks,
                                     reserve=ledger.feed_reserved)
            orders.extend(sell)
            if not in_endgame:
                hire_slots = max(0, max_orders - len(sell))
                target = policy.get("target_hands", 10) if isinstance(policy, dict) else 10
                for _ in range(min(target, hire_slots)):
                    orders.append(["HIRE"])
        if not in_endgame:
            orders.extend(self.generate_market_orders(
                state, rules, shed_usage=ledger.shed_projected, ledger=ledger))
        elif hour != 0:
            orders.extend(self._sell_orders(shed, endgame=True, rules=rules,
                                            market_stocks=market_stocks,
                                            reserve=ledger.feed_reserved))
        return orders

    # ------------------------------------------------------------------ #
    # Worker assignment
    # ------------------------------------------------------------------ #
    def assign_tasks(self, workers, tasks, rules, shed_usage=0, drop_mode=None):
        """Match workers to tasks and return ``{worker_id: action_list}``.

        Order of operations mirrors the handover's per-hour ladder:
            1. DROP pre-emption -- a worker whose carried load plus current shed
               usage exceeds the active threshold is pulled out of band
               assignment entirely and routed to the nearest shed tile.
            2. Same-day watering tasks are deferred while their PLANT is still
               pending this turn (a seed cannot be watered before it exists).
            3. Remaining workers are matched to remaining tasks band-by-band via
               greedy minimum-Manhattan-distance pairing.
            4. Any still-idle worker returns PASS (an empty list).
        """
        mode = drop_mode if drop_mode is not None else self.drop_mode
        threshold = self._drop_threshold(rules, mode)

        actions = {}
        free = []

        # 1) DROP pre-emption ------------------------------------------------
        # Track a projected shed fill so that when multiple carriers are
        # preempted in the same pass, each successive check sees the load the
        # earlier droppers will add -- otherwise two workers each carrying 20
        # with the shed at 70 both see 90 > 80, both DROP, and the shed hits
        # 110 (overflow / spill).
        projected_shed = shed_usage
        for worker in workers:
            pos = tuple(worker["pos"])
            carried = worker.get("carried", 0)
            # A worker fetching/transporting a task item (wheat for FEED, an
            # animal for PLACE) must not be diverted to DROP -- that would dump
            # the very item it just picked up. Only loose harvest triggers DROP.
            if (carried > 0 and not worker.get("carrying_item")
                    and (carried + projected_shed) > threshold):
                actions[worker["id"]] = self._shed_action(pos)
                projected_shed += carried
            else:
                free.append(worker)

        # 1.5) Carry-lock (#9): a worker already carrying a fetch item is pinned
        # to the nearest task that needs exactly that item and removed from band
        # matching, so a higher-priority band cannot pull it away mid-transport
        # and strand the carried animal / wheat.  Only fires when a task
        # actually wants the carried item; a carried item matching no task falls
        # through to normal matching (where _task_action's wrong-item
        # DROP-before-PICKUP rule handles it).  ``carrying_item`` is only ever
        # set in the animal fetch/feed flows, so this is inert with animals off.
        locked_tasks = set()
        still_free = []
        for worker in free:
            item = worker.get("carrying_item")
            if not item:
                still_free.append(worker)
                continue
            wpos = tuple(worker["pos"])
            best = None
            for i, t in enumerate(tasks):
                if i in locked_tasks or t.fetch != item:
                    continue
                d = manhattan(wpos, t.pos)
                if best is None or d < best[0]:
                    best = (d, i, t)
            if best is not None:
                locked_tasks.add(best[1])
                actions[worker["id"]] = self._task_action(worker, best[2])
            else:
                still_free.append(worker)
        free = still_free

        # 2) Re-band same-day watering just below PLANT ----------------------
        # Same-day water tasks were created at priority_dying (9500) so they
        # would outrank other work, but they must be assigned *after* the
        # plant worker has been dispatched (you can't water bare ground).
        # Instead of filtering them out entirely (the old bug), we demote
        # them to just below PLANT (6000) so a *different* worker is
        # assigned to water the newly planted tile on the same turn.
        plant_priority = rules.get("daily_routines", {}).get("PLANT", {}).get("priority", 6000)
        plant_cells = {t.pos for t in tasks if t.kind == "PLANT"}
        active = []
        for i, t in enumerate(tasks):
            if i in locked_tasks:
                continue  # already claimed by a carrying worker (#9)
            if t.same_day and t.pos in plant_cells:
                # Demote to just below PLANT so it's assigned after planting.
                active.append(Task(t.kind, t.pos, plant_priority - 1,
                                   value=t.value, crop=t.crop,
                                   same_day=t.same_day, item=t.item,
                                   fetch=t.fetch))
            else:
                active.append(t)

        # 3) Optimal Manhattan matching, band by band -----------------------
        # Bands are grouped by *priority only* (#22/#23): every task sharing an
        # urgency band competes for the same workers, and their relative
        # economic value is folded into the assignment cost inside _match_band
        # rather than split into value sub-bands.  A short-handed crew then
        # serves the high-value tasks in a band and skips the low-value ones,
        # while value never crosses an urgency boundary (priority still wins).
        active.sort(key=Task.sort_key, reverse=True)
        value_weight = rules.get("routing_policy", {}).get("value_weight", 0.0)
        idx, count = 0, len(active)
        while idx < count and free:
            band_priority = active[idx].priority
            band = []
            while idx < count and active[idx].priority == band_priority:
                band.append(active[idx])
                idx += 1
            self._match_band(free, band, actions, value_weight=value_weight)

        # 4) Idle workers pass ----------------------------------------------
        for worker in free:
            actions.setdefault(worker["id"], [])

        return actions

    def _match_band(self, free, band, actions, value_weight=0.0):
        """Assign workers to tasks within one band by optimal minimum-cost
        bipartite matching (Hungarian algorithm).

        Minimises the total cost across all worker-task pairs in the band.  The
        base cost is Manhattan distance; when ``value_weight`` > 0 an economic
        term ``value_weight * (vmax - task.value)`` is added so higher-value
        tasks are cheaper (#22/#23).  Crucially this term is a *per-task
        column constant*: when workers >= tasks every task is assigned exactly
        once, so the added constants sum the same for every complete matching
        and the optimal assignment is identical to pure distance.  It changes
        the result only when the band is short-handed (tasks > workers), where
        it decides *which* low-value tasks get left undone this turn -- exactly
        the delay-consequence sub-ranking the manager asked for.

        The term stays non-negative (``value_weight`` >= 0 and ``vmax`` is the
        band maximum, so ``vmax - value`` >= 0), which the ``_hungarian`` BIG
        sentinel requires.  Pure Python (no scipy/numpy).  Mutates ``free``
        (removing assigned workers) and ``actions``.
        """
        if not free or not band:
            return

        nw = len(free)
        nt = len(band)

        # Build the cost matrix.  Rows = workers, cols = tasks.  Base term is
        # Manhattan distance; the value fold (see docstring) breaks ties toward
        # richer/more-urgent tasks under contention without ever going negative.
        vmax = max((t.value for t in band), default=0.0)
        costs = [[manhattan(tuple(free[wi]["pos"]), band[ti].pos)
                  + value_weight * (vmax - band[ti].value)
                  for ti in range(nt)] for wi in range(nw)]

        # The Hungarian algorithm finds the minimum-cost assignment.  For the
        # small matrices here (≤11 workers × ~20 tasks) the O(n^3) cost is
        # negligible.  We use the rectangular variant: pad to square with
        # large costs so extra workers/tasks are left unassigned.
        assignment = _hungarian(costs)

        used_workers = set()
        used_tasks = set()
        for wi, ti in assignment:
            worker = free[wi]
            actions[worker["id"]] = self._task_action(worker, band[ti])
            used_workers.add(wi)
            used_tasks.add(ti)

        free[:] = [w for i, w in enumerate(free) if i not in used_workers]

    # ------------------------------------------------------------------ #
    # Action helpers
    # ------------------------------------------------------------------ #
    def _task_action(self, worker, task):
        """Emit the move-or-work action for ``worker`` assigned to ``task``."""
        pos = tuple(worker["pos"])
        if task.kind == "DROP":
            return self._shed_action(pos)
        # Fetch step: FEED/PLACE need an item carried from the shed first. Route
        # the worker to a shed tile and PICKUP before heading to the work tile.
        if task.fetch and worker.get("carrying_item") != task.fetch:
            # If carrying the *wrong* item, DROP it first -- the engine does
            # not allow PICKUP while already holding something.
            if worker.get("carrying_item") is not None:
                if pos in SHED_TILES:
                    return ["DROP"]
                return step_towards(pos, nearest_shed_tile(pos))
            if pos in SHED_TILES:
                return ["PICKUP", task.fetch]
            return step_towards(pos, nearest_shed_tile(pos))
        if pos == task.pos:
            return task.action()
        return step_towards(pos, task.pos)

    def _shed_action(self, pos):
        """DROP if standing on a shed tile, else step toward the nearest one."""
        if pos in SHED_TILES:
            return ["DROP"]
        return step_towards(pos, nearest_shed_tile(pos))

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_repeater(crop, rules, ctype):
        """Whether a crop is a repeater (tomato/strawberry)."""
        flagged = crop.get("repeater")
        if flagged is not None:
            return bool(flagged)
        params = rules.get("crop_params", {}).get(ctype)
        return bool(params) and params.get("type") == "repeater"

    @staticmethod
    def _build_action(home_type):
        """Map an animal's home type to the BUILD verb that raises it.

        GOOSE lives in a COOP (``BUILD_COOP``); COW/SHEEP share a PASTURE
        (``BUILD_PASTURE``). Returns None for an unknown home type so an
        unrecognised animal simply produces no build task.
        """
        return {"COOP": "BUILD_COOP", "PASTURE": "BUILD_PASTURE"}.get(home_type)

    @staticmethod
    def _water_priority(crop, water_bands, to_die, is_repeater):
        """Pick the watering band for a crop, or None if it needs no water.

        A plant one mark short of death always wins the top band, regardless of
        whether it was already watered this period -- that is the whole point of
        the 9500 band. Otherwise a task is only raised when the crop still needs
        water this period (``needs_water``):
            * one-time crop inside its yield-bonus window -> bonus (8000)
            * repeater (tomato / strawberry)              -> ongoing (7000)
            * anything else                               -> normal (5000)
        """
        if crop.get("misses", 0) >= to_die - 1:
            return water_bands["priority_dying"]
        if not crop.get("needs_water", False):
            return None
        if crop.get("in_bonus_window", False):
            return water_bands["priority_bonus"]
        if is_repeater:
            return water_bands["priority_ongoing"]
        return water_bands["priority_normal"]

    @staticmethod
    def _crop_value(rules, ctype, market_stocks):
        """Market value of a crop, used as the HARVEST in-band tie-break.

        Uses the live single-unit market price at the current stock so a crop
        whose market has already crashed is not over-prioritized.
        """
        stock = market_stocks.get(ctype, 0) if market_stocks else 0
        return float(MarketModel.market_price(rules, ctype, stock))

    @staticmethod
    def _fertilizer_buy_qty(obs, rules, shed_usage):
        """How many FERTILIZER units to buy this turn (0 = none).

        Fertilizer is worth buying only when standing crops would actually
        yield more with it (``yield_fertilized > yield_no_fertilizer`` -- true
        for wheat/carrot, not melon) and we do not already hold enough in the
        shed. The order is clamped to the free shed slots so a purchase can
        never push the shed past ``shed_size``: bought fertilizer occupies a
        shed slot until a worker picks it up to apply.
        """
        crops = obs.get("crops") or []
        shed = obs.get("shed") or {}
        shed_size = rules["constants"]["shed_size"]
        crop_params = rules.get("crop_params", {})

        beneficial = 0
        for crop in crops:
            params = crop_params.get(crop.get("type"))
            if not params:
                continue
            if params.get("yield_fertilized", 0) > params.get("yield_no_fertilizer", 0):
                beneficial += 1
        if beneficial == 0:
            return 0

        have = shed.get("FERTILIZER", 0) or 0
        need = max(0, beneficial - have)
        capacity_left = max(0, shed_size - shed_usage)
        return min(need, capacity_left)

    def _choose_plant_crop(self, rules, step, market_stocks):
        """Best-EV crop that can still be harvested and sold before the season
        ends, or None if nothing passes the cash test (then we plant nothing).

        Memoized per (step, market-stock signature) (#12): the crop-EV ranking
        and the cash test are pure functions of the turn's step and market
        stocks, so re-entering ``generate_tasks`` within a turn should not repeat
        the ``crop_values`` sort.  The cache holds a single (key, result) pair;
        a new turn's key misses and recomputes, so the result is never stale."""
        stocks = market_stocks or {}
        key = (step, tuple(sorted(stocks.items())))
        cached = self._plant_choice_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        ranked = MarketModel.crop_values(rules, stocks)
        result = None
        for ctype in ranked:
            if self._can_finish(ctype, rules, step):
                result = ctype
                break
        self._plant_choice_cache = (key, result)
        return result

    @staticmethod
    def _can_finish(ctype, rules, step):
        """Planting cash test: can this crop mature (and leave time to sell)
        before turn 720?

        Cutoff day = ``season_days - grow_days - sell_days_buffer``.  The
        sell buffer is read from ``constants.sell_days_buffer`` (default 1, the
        handover value) rather than a hard-coded ``-1`` (#24), so the days
        reserved to liquidate the final harvest are tunable in data alone.  At
        the default buffer of 1 this reproduces the handover's carrot 26 /
        wheat 25 / melon 19 cutoffs exactly."""
        params = rules.get("crop_params", {}).get(ctype)
        if not params:
            return False
        phase = turn_phase(step, rules)
        season_days = phase["season_days"]
        day = phase["day"]
        sell_buffer = rules.get("constants", {}).get("sell_days_buffer", 1)
        if params.get("type") == "repeater":
            grow_days = params.get("first_fruit", season_days)
        else:
            grow_days = params.get("full_harvest", season_days)
        return day <= season_days - grow_days - sell_buffer

    def _drop_threshold(self, rules, mode):
        """Item count above which a carrying worker is sent to the shed.

        normal   -> policy.drop_pressure of the shed (default 0.8 -> 80 items)
        overflow -> half the shed (tighten when a spill is imminent)
        endgame  -> 0 (every carried item must be banked before the clock ends)
        """
        shed_size = rules.get("constants", {}).get("shed_size", 100)
        if mode == "endgame":
            return 0
        if mode == "overflow":
            return 0.5 * shed_size
        return rules.get("policy", {}).get("drop_pressure", 0.8) * shed_size
