import calendar
import datetime
import sys

# Handle holidays package
try:
    import holidays
except ImportError:
    print("This script requires 'holidays' for handling public holidays.")
    print("Install it using: pip install holidays")
    sys.exit(1)

# Handle PuLP package
try:
    import pulp
except ImportError:
    print("This script requires 'pulp' (PuLP) for optimization.")
    print("Install it using: pip install pulp")
    sys.exit(1)
    
def _last_day_of_month(year, month):
    """Return a date object representing the last day of the given year and month."""
    if month == 12:
        return datetime.date(year, 12, 31)
    return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

def get_japanese_holidays_in_range(start_year, start_month, end_year, end_month):
    """
    Return a set of date objects for Japanese holidays within the specified month range.
    """
       
    holiday_set = set()
    for y in range(start_year, end_year + 1):
        jp_holidays = holidays.Japan(years=[y])
        for day in jp_holidays.keys():
            if (datetime.date(y, start_month, 1) <= day <= _last_day_of_month(end_year, end_month)):
                holiday_set.add(day)
    return holiday_set

def _last_day_of_month(year, month):
    """Return a date object of the last day of the given year/month."""
    if month == 12:
        return datetime.date(year, 12, 31)
    return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

def generate_date_list(start_year, start_month, end_year, end_month):
    """
    Generate all dates from the start of (start_year, start_month)
    through the end of (end_year, end_month).
    """
    start_date = datetime.date(start_year, start_month, 1)
    end_date = _last_day_of_month(end_year, end_month)
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += datetime.timedelta(days=1)

    return all_dates

def plan_workdays_multiple_months(
    start_year, start_month, end_year, end_month,
    disabled_days=None, forced_days=None, want_overtime=False
):
    """
    Solves the scheduling problem across multiple consecutive months:

    - Exactly 10 workdays per calendar month (in each month from start to end).
    - A 'workday' has at least 7.75 hours (no explicit daily max).
    - No more than 28 hours in any rolling 7-day window (across month boundaries).
    - If want_overtime=True, maximize total hours; otherwise, minimize.
    - 'forced_days': set of specific dates that MUST be workdays.

    Returns a dict {date: assigned_hours, ...} for each date in the range,
    or an empty dict if no feasible solution is found.
    """
    if disabled_days is None:
        disabled_days = set()
    else:
        disabled_days = set(disabled_days)

    if forced_days is None:
        forced_days = set()
    else:
        forced_days = set(forced_days)

    # Generate the overall date range
    date_list = generate_date_list(start_year, start_month, end_year, end_month)
    holiday_set = get_japanese_holidays_in_range(start_year, start_month, end_year, end_month)

    # Prepare the LP problem
    problem_sense = pulp.LpMaximize if want_overtime else pulp.LpMinimize
    prob = pulp.LpProblem("MultiMonthWorkdayPlanner", problem_sense)

    # Decision variables:
    # - yes_no[d] in {0,1} indicates if day d is a chosen workday
    # - hours[d] >= 0 indicates how many hours we work on day d
    yes_no = {}
    hours = {}

    for d in date_list:
        feasible = not (
            d.weekday() >= 5 or  # Sat=5, Sun=6
            d in holiday_set or
            d in disabled_days
        )

        var_name = d.isoformat()

        # If forced, then yes_no must be 1, otherwise up to feasibility
        if d in forced_days:
            # If day is forced but also a weekend/holiday/disabled, likely infeasible.
            # We'll let the solver handle or fail, but we can also do a quick check.
            if not feasible:
                print(f"[!] Forced day {d} is not feasible (weekend/holiday/disabled). Could be infeasible.")
            yes_no_ub = 1
        else:
            yes_no_ub = 1 if feasible else 0
        
        if not feasible:
            # Create a normal binary variable
            yes_no[d] = pulp.LpVariable(f"yes_no_{var_name}", cat=pulp.LpBinary)

            # Then force it to 0 via a constraint
            prob += (yes_no[d] == 0, f"ForceZero_{d}")
        else:
            yes_no[d] = pulp.LpVariable(f"yes_no_{var_name}", cat=pulp.LpBinary, upBound=1)
            
        # Modify the hours variable to explicitly cap daily hours
        hours[d] = pulp.LpVariable(
            f"hours_{var_name}",
            lowBound=0,  # Minimum hours: 0
            upBound=14  # Maximum hours: 14
        )

    # Constraint 1: Exactly 10 workdays per month
    all_ym = sorted(list({(d.year, d.month) for d in date_list}))
    for (y, m) in all_ym:
        dates_in_month = [d for d in date_list if (d.year, d.month) == (y, m)]
        prob += (
            pulp.lpSum([yes_no[d] for d in dates_in_month]) == 10,
            f"Exactly10Days_{y}_{m}"
        )

    # Constraint 2: If yes_no[d] = 1, then hours[d] >= 7.75.
    # If yes_no[d] = 0, then hours[d] = 0.
    for d in date_list:
        prob += (hours[d] >= 7.75 * yes_no[d], f"Min7_75_ifWorked_{d}")
        prob += (hours[d] <= 14, f"Max14HoursIfWorked_{d}")  # Enforce 14-hour max
        prob += (hours[d] <= 28 * yes_no[d], f"Zero_ifNotWorked_{d}")

    # Constraint 2b: forced_days must be chosen (yes_no=1).
    for fd in forced_days:
        prob += (yes_no[fd] == 1, f"ForcedDay_{fd}")

    # Constraint 3: No more than 28 hours in any rolling 7-day window
    date_list_sorted = sorted(date_list)
    for i, d in enumerate(date_list_sorted):
        window_end = d + datetime.timedelta(days=6)
        window_days = [x for x in date_list_sorted if d <= x <= window_end]
        prob += (
            pulp.lpSum([hours[x] for x in window_days]) <= 28,
            f"Max28h_7days_start_{d}"
        )

    # Objective: sum of hours over entire range
    total_hours = pulp.lpSum([hours[d] for d in date_list])
    prob.setObjective(total_hours)

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    if status_str not in ("Optimal", "Feasible"):
        print(f"[!] No feasible solution found. Solver status: {status_str}")
        return {}

    # Collect results
    schedule = {}
    for d in date_list:
        val = pulp.value(hours[d])
        if val is None:
            val = 0.0
        schedule[d] = round(val, 2)

    return schedule

import calendar
import datetime

def parse_short_date_input(line: str) -> set:
    """
    Parse user input like '2025-2-1,2,5,3-25,20,2026-1-15' into date objects.
    Raises ValueError if any token is malformed or out of range.
    """
    tokens = [t.strip() for t in line.split(",") if t.strip()]
    result = set()

    current_year = None
    current_month = None

    for tok in tokens:
        parts = tok.split("-")
        if len(parts) == 3:
            # format: YYYY-mm-dd
            y, m, d = parts
            y_i = int(y)
            m_i = int(m)
            d_i = int(d)
            # Validate
            _check_valid_date(y_i, m_i, d_i)
            current_year = y_i
            current_month = m_i
            result.add(datetime.date(y_i, m_i, d_i))

        elif len(parts) == 2:
            # Could be (year, month) or (month, day).
            p1, p2 = parts
            p1_i = int(p1)
            p2_i = int(p2)
            # Decide which is year vs month vs day
            if p1_i > 12:
                # We interpret p1 as year, p2 as month => year-month
                _check_year_month(p1_i, p2_i)
                current_year = p1_i
                current_month = p2_i
            else:
                if current_year is None:
                    # interpret p1 as year, p2 as month
                    _check_year_month(p1_i, p2_i)
                    current_year = p1_i
                    current_month = p2_i
                else:
                    # interpret p1 as month, p2 as day => must have year
                    _check_valid_date(current_year, p1_i, p2_i)
                    current_month = p1_i
                    result.add(datetime.date(current_year, current_month, p2_i))
        elif len(parts) == 1:
            # Just a day number?
            d_i = int(parts[0])
            if current_year is not None and current_month is not None:
                _check_valid_date(current_year, current_month, d_i)
                result.add(datetime.date(current_year, current_month, d_i))
            else:
                # We have no prefix => error
                raise ValueError(f"Cannot parse single day '{tok}' without a known prefix.")
        else:
            raise ValueError(f"Could not parse token '{tok}'")

    return result

def _check_year_month(y: int, m: int):
    """Raise ValueError if invalid year-month."""
    if m < 1 or m > 12:
        raise ValueError(f"Invalid month={m} (should be 1..12)")
    if y < 1 or y > 9999:
        raise ValueError(f"Invalid year={y}")

def _check_valid_date(y: int, m: int, d: int):
    """Use datetime.date() to ensure validity or raise ValueError."""
    datetime.date(y, m, d)  # Will raise ValueError if out of range

def safe_parse_short_date_input(prompt: str) -> set:
    """
    Prompt user repeatedly until they provide a valid short-date line or press Enter (returns empty set).
    Any invalid token in the line => entire line is discarded => user re-prompts.
    """
    while True:
        line = input(prompt).strip()
        if not line:
            return set()  # empty => no dates
        try:
            parsed = parse_short_date_input(line)
            return parsed
        except ValueError as e:
            print(f"Error parsing '{line}': {e}")
            print("Please re-enter the entire list of dates correctly.\n")



# The combined display function:

def display_schedule(
    schedule,
    start_year, start_month,
    end_year, end_month,
    disabled_days=None,
    forced_days=None
):
    """
    Prints:
      1) A table of all dates with columns:
         - Date
         - WkDay
         - Type  (weekend, holiday, forbidden, forced, working, free)
         - Hours
         - Sum_of_next_7d

      2) A color-coded monthly calendar view.
    """
    if disabled_days is None:
        disabled_days = set()
    if forced_days is None:
        forced_days = set()

    holiday_set = get_japanese_holidays_in_range(start_year, start_month, end_year, end_month)

    # Colors
    COLOR_LIGHT_BLUISH_GREEN = "\033[38;5;86m"  # forced
    COLOR_LIGHT_RED          = "\033[91m"       # weekend & holiday
    COLOR_MAGENTA            = "\033[38;5;207m" # forbidden
    COLOR_GREEN              = "\033[92m"       # working
    COLOR_WHITE              = "\033[37m"       # free
    COLOR_RESET              = "\033[0m"

    def get_day_type(d, hrs):
        wday = d.weekday()
        if d in forced_days:
            return "forced"
        if wday >= 5:
            return "weekend"
        if d in holiday_set:
            return "holiday"
        if d in disabled_days:
            return "forbidden"
        if hrs > 0:
            return "working"
        return "free"

    def get_color(day_type):
        if day_type == "forced":
            return COLOR_LIGHT_BLUISH_GREEN
        if day_type == "working":
            return COLOR_GREEN
        if day_type in ("weekend", "holiday"):
            return COLOR_LIGHT_RED
        if day_type == "forbidden":
            return COLOR_MAGENTA
        return COLOR_WHITE

    # Table
    date_list = generate_date_list(start_year, start_month, end_year, end_month)
    date_list_sorted = sorted(date_list)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    print("Date        WkDay    Type     Hours  Sum_of_next_7d")
    print("-----------------------------------------------------")

    for d in date_list_sorted:
        end_d = d + datetime.timedelta(days=6)
        window_days = [x for x in date_list_sorted if d <= x <= end_d]
        sum_7d = sum(schedule.get(x, 0) for x in window_days)
        hrs = schedule.get(d, 0.0)
        dt = get_day_type(d, hrs)
        clr = get_color(dt)

        row = (
            f"{d}  {weekdays[d.weekday()]:>3}   "
            f"{clr}{dt:<10}{COLOR_RESET} "
            f"{hrs:5.2f}   "
            f"{sum_7d:5.2f}"
        )
        print(row)

    # Calendar
    print("\n========== Calendar View (Color-Coded) ==========")

    def month_range(y1,m1,y2,m2):
        y,m = y1,m1
        while (y<y2) or (y==y2 and m<=m2):
            yield y,m
            m+=1
            if m>12:
                m=1
                y+=1

    for (yr, mo) in month_range(start_year, start_month, end_year, end_month):
        title = f"{calendar.month_name[mo]} {yr}"
        print("\n"+title.center(29," "))
        print(" Su Mo Tu We Th Fr Sa")
        cal = calendar.Calendar(firstweekday=6)
        weeks = []
        cur = []
        for day in cal.itermonthdates(yr, mo):
            if day.month!=mo:
                cur.append("  ")
            else:
                day_hrs = schedule.get(day,0.0)
                dt = get_day_type(day, day_hrs)
                c = get_color(dt)
                ds = f"{day.day:2d}"
                ds_col = f"{c}{ds}{COLOR_RESET}"
                cur.append(ds_col)
            if len(cur)==7:
                weeks.append(cur)
                cur=[]
        if cur:
            weeks.append(cur)
        for w in weeks:
            print(" ".join(w))
     
    legend = {
        "working": ( get_color("working"), "Working (Scheduled normal workdays)"),
        "weekend": ( get_color("weekend"), "Weekend (Saturdays and Sundays)"),
        "holiday": (get_color( "holiday"), "Holiday (National holidays)"),
        "forbidden": ( get_color("forbidden"), "Forbidden (User-disabled days)"),
        "forced": (get_color("forced"), "Forced (User-mandated workdays)"),
        "free": ( get_color("free"), "Free (Non-working days)"),
    }

    print("\nColor Legend:")
    for key, (color, description) in legend.items():
        print(f"{color}■\033[0m {description}")

    print("\n=========================================\n")


def main():
    """
    1) Prompt for start date & duration
    2) Solve w/ no forced/disabled -> show initial display
    3) Repeatedly let user modify forced/disabled via short input
    4) Re-solve & display.
    """
    print("====== Multi-Month Workday Planner ======")

    today = datetime.date.today()
    next_month = today.replace(day=1) + datetime.timedelta(days=32)
    default_start_year = next_month.year
    default_start_month = next_month.month

    # Prompt start
    sy_str = input(f"Enter start year (e.g., {default_start_year}) [default: {default_start_year}]: ")
    sm_str = input(f"Enter start month (1-12) [default: {default_start_month}]: ")

    try:
        sy = int(sy_str) if sy_str.strip() else default_start_year
        sm = int(sm_str) if sm_str.strip() else default_start_month
    except ValueError:
        print("Invalid input. Using default start date.")
        sy, sm = default_start_year, default_start_month

    # duration
    duration_str = input("Enter duration in months (e.g., 2) [default: 2]: ")
    try:
        duration = int(duration_str) if duration_str.strip() else 2
        if duration<1:
            raise ValueError
    except ValueError:
        duration=2
        print("Invalid. Using default: 2 months.")

    # compute end
    start_date = datetime.date(sy, sm, 1)
    end_date = start_date + datetime.timedelta(days=31*(duration-1))
    ey, em = end_date.year, end_date.month

    print(f"Planning schedule from {sy}-{sm:02d} to {ey}-{em:02d} ({duration} months).")

    # Overtime?
    ot_pref = input("Maximize overtime? (y/n) [n]: ")
    want_overtime = (ot_pref.lower()=="y")

    # 1) Solve w/ no forced or disabled => initial
    schedule_init = plan_workdays_multiple_months(
        sy, sm, ey, em,
        disabled_days=set(),
        forced_days=set(),
        want_overtime=want_overtime
    )
    if not schedule_init:
        print("No valid schedule found even with no forced/disabled.")
    else:
        print("\n=== Initial schedule (no forced/disabled) ===")
        display_schedule(schedule_init, sy, sm, ey, em,
                        disabled_days=set(), forced_days=set())

    ans = input("Do you want to add forced or disabled working day constraints (y/n) [n]: ").strip()
    if ans.lower() != "y":
       print("Done. Exiting.")
       exit(0)
    # loop editing constraints
    while True:
        print("\nAdd/modify constraints (forced or disabled days).")
        # parse short input for disabled
        new_disabled = safe_parse_short_date_input("Enter disabled days (eg '2025-2-1,2,5,3-25'):\n> ")

        disabled_days = set()
        forced_days = set()
        # we add them
        for nd in new_disabled:
            disabled_days.add(nd)

        # parse short input for forced
        new_forced = safe_parse_short_date_input("Enter forced days (eg '2025-3-1,2,4-25,2025-5-2'):\n> ")

        for nf in new_forced:
            forced_days.add(nf)
        
        # re-solve
        schedule = plan_workdays_multiple_months(
            sy, sm, ey, em,
            disabled_days=disabled_days,
            forced_days=forced_days,
            want_overtime=want_overtime
        )
        if not schedule:
            print("No valid schedule with these constraints.")
        else:
            display_schedule(schedule, sy, sm, ey, em,
                            disabled_days=disabled_days,
                            forced_days=forced_days)

        ans = input("Edit again? (y/n) [n]: ").strip()
        if ans.lower() != "y":
            print("Done. Exiting.")
            break

if __name__=="__main__":
    main()


