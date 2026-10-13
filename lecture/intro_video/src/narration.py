"""Narration script for the lecture intro video.

Each scene is a list of (line, gap) pairs; a line is spoken as one TTS
request and `gap` is the pause after it. Inline markers like [[name]] are
stripped before synthesis; they mark the word that follows so visuals can be
keyed to the moment it is spoken.

`lead` is the visual-only time before the first line, `tail` the time
held after the last line.
"""

VOICE = "en-GB-RyanNeural"
RATE = "+6%"

SCENES = [
    {
        "id": "open",
        "lead": 3.0,
        "tail": 4.4,
        "lines": [
            ("Every time you ask an AI a question, stream a film, or tap to pay, "
             "a building somewhere does the work.", 0.45),
            ("Inside, [[rows]]row after row of servers turn electricity into "
             "computation, and almost all of it, into [[heat]]heat.", 0.5),
            ("These are [[dc]]data centres, and they're becoming one of the most "
             "important loads on the power system.", 0.2),
        ],
    },
    {
        "id": "anatomy",
        "lead": 0.9,
        "tail": 1.2,
        "lines": [
            ("[[grid]]Power arrives from the grid, through the [[ups]]UPS: batteries "
             "that bridge the gap until [[gen]]generators start, if the grid ever "
             "fails.", 0.4),
            ("It feeds [[it]]thousands of servers packed into racks. Virtually every "
             "watt becomes heat, removed by [[cool]]chillers, with servers breathing in "
             "from a [[cold]]cold aisle, and out into a [[hot]]hot aisle.", 0.45),
            ("The overhead is captured by [[pue]]PUE: total facility energy, over IT "
             "energy. The average is [[avg]]about one point five. The best sites, "
             "[[best]]around one point one.", 0.2),
        ],
    },
    {
        "id": "scale",
        "lead": 0.7,
        "tail": 1.2,
        "lines": [
            ("Demand is soaring. Data centres used around [[y2025]]four hundred and "
             "eighty-five terawatt-hours in 2025, and the IEA expects [[y2030]]roughly "
             "double that by 2030: about as much as [[japan]]Japan uses today.", 0.45),
            ("The driver is [[ai]]AI. A conventional rack draws [[old]]five to ten "
             "kilowatts. An AI rack, [[new]]well over a hundred: as much as "
             "[[homes]]four hundred British homes. That's why [[liquid]]liquid cooling "
             "is going mainstream.", 0.2),
        ],
    },
    {
        "id": "britain",
        "lead": 1.2,
        "tail": 1.2,
        "lines": [
            ("In Britain, data centres already draw [[now]]around two point four "
             "gigawatts, clustered around [[london]]London. In [[west]]West London, the "
             "grid grew so congested that new homes were warned they might wait years "
             "for a connection.", 0.45),
            ("Yet developers want to connect [[queue]]more than fifty gigawatts of new "
             "data centres: more than Britain's [[peak]]peak demand. And the government "
             "has named [[zones]]AI Growth Zones, from [[lanark]]Lanarkshire to "
             "[[swales]]South Wales.", 0.45),
            ("Meanwhile, the cleanest power comes from [[wind]]wind in Scotland and the "
             "North Sea. On the windiest days, wind farms are [[curtail]]paid to switch "
             "off, because the network can't carry it all south.", 0.4),
            ("So [[where]]where we build data centres, and [[when]]when they run, "
             "really matters.", 0.2),
        ],
    },
    {
        "id": "prices",
        "lead": 0.8,
        "tail": 1.0,
        "lines": [
            ("[[carpet]]This is every quarter-hour of Britain's day-ahead power price "
             "in 2025: each column a day, each row a time of day.", 0.4),
            ("The [[evening]]hot bands are evening peaks, set by gas. The [[cheap]]cool "
             "stretches are windy hours, and in [[negative]]six hundred and seventy-two "
             "quarter-hours, prices fell below zero.", 0.4),
            ("A load that could follow this pattern would be [[valuable]]enormously "
             "valuable.", 0.2),
        ],
    },
    {
        "id": "toolkit",
        "lead": 0.5,
        "tail": 0.4,
        "lines": [
            ("So how do we make data centres low-carbon? [[eff]]Efficiency, with free "
             "and liquid cooling. [[clean]]Clean supply, from wind and solar, "
             "increasingly matched [[hourly]]hour by hour.", 0.4),
            ("[[heat]]Heat reuse: in [[odense]]Denmark, one data centre warms thousands "
             "of homes, and London is planning a network at [[oldoak]]Old Oak.", 0.4),
            ("[[flex]]And flexibility.", 0.2),
        ],
    },
    {
        "id": "flex",
        "lead": 0.4,
        "tail": 1.0,
        "lines": [
            ("The grid must balance supply and demand [[balance]]every second: a job "
             "that used to fall to [[fossil]]fossil-fuelled power stations. With wind "
             "and sun, [[demand]]demand must flex too. And data centres have three "
             "levers.", 0.45),
            ("[[work]]The workload. A video call must run now, but AI training or a "
             "backup can often [[wait]]wait, [[shift]]shifting computing to when power "
             "is plentiful.", 0.4),
            ("[[batt]]The UPS batteries, which can respond [[ms]]almost instantly.", 0.35),
            ("[[therm]]And the cooling. Chilled water and cold air let chillers ease "
             "off, [[drift]]without servers overheating.", 0.45),
            ("It's already happening. In a [[trial]]London trial with National Grid, "
             "an AI cluster cut its power by [[forty]]up to forty percent on grid "
             "signals. And [[google]]Google has agreed to curtail around a gigawatt of "
             "US demand, when grids are stressed.", 0.2),
        ],
    },
    {
        "id": "close",
        "lead": 0.5,
        "tail": 5.2,
        "lines": [
            ("Data centres can be a [[strain]]strain on the grid, or one of its most "
             "[[asset]]valuable flexible assets. It depends on how we [[design]]design, "
             "[[build]]site, and [[operate]]operate them.", 0.5),
            ("[[q1]]How much flexibility can they offer? [[q2]]For how long? "
             "[[q3]]At what cost?", 0.5),
            ("[[go]]Let's find out.", 0.0),
        ],
    },
]
