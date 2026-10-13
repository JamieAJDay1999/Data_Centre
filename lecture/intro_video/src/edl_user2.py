"""Edit decision list for the second presenter recording (Recording 41).

Pauses are kept as recorded (no tightening). Only retakes and the
"Lancashire" slip are removed.
"""

# Boundary modes: "s" onset, "e" offset, "c" local minimum, "x" exact.
# ("gapfrom", t) marks where the recorded pause after the line really starts
# (used when the end of the line was cut out).
EDL = {
    "open": [
        [("seg", 0.75, 7.13, "s", "e")],
        [("seg", 8.15, 14.50, "s", "e")],
        [("seg", 15.85, 20.04, "s", "e")],
    ],
    "anatomy": [
        [("seg", 22.90, 30.17, "s", "e")],
        [("seg", 31.20, 43.10, "s", "e")],
        [("seg", 44.60, 54.97, "s", "e")],
    ],
    "scale": [
        # "Demand is soaring." from the first take, then the clean retake from "Data centres used...".
        [("seg", 57.90, 58.87, "x", "e"), ("pause", 1.00), ("seg", 68.35, 79.62, "x", "e")],
        [("seg", 81.25, 94.66, "s", "e")],
    ],
    "britain": [
        [("seg", 97.20, 109.53, "s", "e")],
        # Ends at "...AI Growth Zones" (drops "from Lancashire to South Wales").
        [("seg", 111.20, 121.30, "s", "x"), ("gapfrom", 123.45)],
        [("seg", 125.15, 135.77, "s", "e")],
        [("seg", 138.90, 142.94, "s", "e")],
    ],
    "prices": [
        [("seg", 145.85, 156.83, "s", "e")],
        [("seg", 158.55, 168.96, "s", "e")],
        [("seg", 171.15, 174.65, "s", "e")],
    ],
    "toolkit": [
        [("seg", 177.70, 188.46, "s", "e")],
        [("seg", 190.05, 197.19, "s", "e")],
        [("seg", 198.85, 200.48, "s", "e")],
    ],
    "flex": [
        [("seg", 201.70, 214.17, "s", "e")],
        [("seg", 215.00, 224.76, "s", "e")],
        [("seg", 227.00, 230.22, "s", "e")],
        [("seg", 230.85, 236.98, "s", "e")],
        [("seg", 238.65, 251.89, "s", "e")],
    ],
    "close": [
        [("seg", 254.00, 262.06, "s", "e")],
        [("seg", 263.35, 267.46, "s", "e")],
        [("seg", 268.35, 269.50, "s", "e")],
    ],
}

CAPTIONS = {
    "open": ["Every time you ask AI a question, stream a film, or tap to pay, a building somewhere does the work.",
             "Inside, row after row of servers turn electricity into computation, and almost all of it is turned into heat.",
             "These are data centres, and they're becoming one of the most important loads on the power system."],
    "anatomy": ["Power arrives from the grid, through the UPS: batteries that bridge the gap until generators start, if the grid ever fails.",
                "It feeds thousands of servers packed into racks. Virtually every watt of this power becomes heat, removed by chillers, with servers breathing in from a cold aisle, and out into a hot aisle.",
                "The overhead is captured by PUE: total facility energy, over IT energy. The average is about 1.5; the best sites, around 1.1."],
    "scale": ["Demand is soaring. Data centres used around 485 terawatt-hours in 2025, and the IEA expects roughly double that by 2030: about as much as Japan uses today.",
              "The driver is AI. A conventional rack draws 5 to 10 kilowatts; an AI rack, well over 100: as much as 400 British homes. That's why liquid cooling is going mainstream."],
    "britain": ["In Britain, data centres already draw 2.4 gigawatts, clustered around London. In West London, the grid grew so congested that new homes were warned they might wait years for a connection.",
                "Yet developers want to connect more than 50 gigawatts of new data centres: more than Britain's peak demand. And the government has named AI Growth Zones.",
                "Meanwhile, the cleanest power comes from Scotland and the North Sea. On the windiest days, wind farms are paid to switch off, because the network can't carry it all south.",
                "So where we build data centres, and when they run, really matters."],
    "prices": ["This is every quarter-hour of Britain's day-ahead power price in 2025. Each column represents a day, each row a time of day.",
               "The hot bands are evening peaks, set by gas. The cool stretches are windy hours, and in 672 quarter-hours, prices fell below zero.",
               "A load that could roughly follow this pattern would be enormously valuable."],
    "toolkit": ["So how do we make data centres low-carbon? Efficiency, with free and liquid cooling; clean supply, from wind and solar, increasingly matched hour by hour.",
                "Heat reuse: in Denmark, one data centre warms thousands of homes, and London is planning a network at Old Oak.",
                "And also flexibility."],
    "flex": ["The grid must balance supply and demand every second: a job that used to fall to fossil-fuel power stations. With wind and sun, demand must flex too, and data centres have three levers.",
             "The workload. A video call must run now, but AI training or a backup can often wait, shifting computing to when power is plentiful.",
             "The UPS batteries, which can respond almost instantly.",
             "And the cooling. Chilled water and cold air let chillers ease off without servers overheating.",
             "And this is already happening. In a London trial with National Grid, an AI cluster cut its power by up to 40% on grid signals. And Google has agreed to curtail around a gigawatt of US demand when grids are stressed."],
    "close": ["Data centres can be the strain on the grid, or one of the most valuable flexible assets. It depends how we design, site and operate them.",
              "How much flexibility can they offer? For how long? And at what cost?",
              "Let's find out now."],
}
