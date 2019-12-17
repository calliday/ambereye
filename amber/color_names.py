# r=g=b: 000, 111, 222, 333
# B>r=g: 001, 002, 003, 112, 113, 223
# r=g>B: 110, 220, 221, 330, 331, 332
# R>g=b: 100, 200, 300, 211, 311, 322
# g=b>R: 011, 022, 033, 122, 133, 233
# G>r=b: 010, 020, 030, 121, 131, 232
# r=b>G: 101, 202, 303, 212, 313, 323
# r>g>b: 210, 310, 320, 321
# r>b>g: 201, 301, 302, 312
# b>r>g: 102, 103, 203, 213
# b>g>r: 012, 013, 023, 123
# g>b>r: 021, 031, 032, 132
# g>r>b: 120, 130, 230, 231
#
# 000 = Black
# 111 = Dark Gray
# 222 = Light Gray
# 333 = White
#
# 001 = Deep Blue
# 002 = Deep Blue
# 003 = Blue
# 112 = Medium Gray-Blue
# 113 = Medium Blue
# 223 = Medium Blue
#
# 110 = Deep Yellow
# 220 = Deep Yellow
# 221 = Medium Gray-Yellow
# 330 = Yellow
# 331 = Medium Yellow
# 332 = Medium Yellow
#
# 100 = Deep Red
# 200 = Deep Red
# 300 = Red
# 211 = Medium Gray-Red
# 311 = Medium Red
# 322 = Medium Red
#
# 011 = Medium Gray-Cyan
# 022 = Deep Cyan
# 033 = Cyan
# 122 = Medium Gray-Cyan
# 133 = Medium Cyan
# 233 = Medium Cyan
#
# 010 = Deep Green
# 020 = Deep Green
# 030 = Green
# 121 = Medium Gray-Green
# 131 = Medium Green
# 232 = Medium Green
#
# 101 = Deep Magenta
# 202 = Deep Magenta
# 303 = Megenta
# 212 = Medium Gray-Magenta
# 313 = Medium Magenta
# 323 = Medium Magenta
#
# 210 = Deep Orange
# 310 = Orange
# 320 = Orange
# 321 = Medium Orange
#
# 201 = Deep Magenta-Red
# 301 = Magenta-Red
# 302 = Magenta-Red
# 312 = Medium Magenta-Red
#
# 102 = Deep Purple
# 103 = Purple
# 203 = Purple
# 213 = Medium Purple
#
# 012 = Deep Royal Blue
# 013 = Royal Blue
# 023 = Royal Blue
# 123 = Medium Royal Blue
#
# 021 = Deep Turquoise
# 031 = Turquoise
# 032 = Turquoise
# 132 = Medium Turquoise
#
# 120 = Deep Lime Green
# 130 = Lime Green
# 230 = Lime Green
# 231 = Medium Lime Green

colors_caleb = [
    [
        ["Black","Dark Blue","Blue","Neon Blue"],
        ["Dark Green","Dark Teal","Dark Blue","Blue"],
        ["Green","Green","Cyan","Light Blue"],
        ["Neon Green","Neon Green","Turquoise","Turquoise"],
    ],
    [
        ["Dark Red","Magenta","Purple","Blue"],
        ["Brown","Dark Gray","Purple","Blue"],
        ["Green","Green","Cyan","Light Blue"],
        ["Neon Green","Neon Green","Light Green","Light Blue"],
    ],
    [
        ["Red","Magenta","Magenta","Purple"],
        ["Dark Orange", "Soft Red","Purple","Purple"],
        ["Yellow","Faded Yellow","Gray","Light Purple"],
        ["Lime Green","Light Lime Green","Light Green","Light Blue"],
    ],
    [
        ["Red","Red","Purple","Purple"],
        ["Orange","Dark Salmon","pink","Magenta"],
        ["Orange","Orange","Faded Pink","Purple"],
        ["Yellow","Yellow","Pale Yellow","White"]
    ]
]

colors_ben = [
    [
        ["Black","Dark Blue","Dark Blue","Blue"],
        ["Dark Green","Dark Green","Blue","Blue"],
        ["Green","Green","Turquoise","Light Blue"],
        ["Neon Green","Neon Green","Turquoise","Light Blue"],
    ],
    [
        ["Dark Red","Purple","Purple","Purple"],
        ["Brown","Dark Gray","Blue","Blue"],
        ["Green","Green","Turquoise","Light Blue"],
        ["Neon Green","Neon Green","Light Green","Light Blue"],
    ],
    [
        ["Dark Red","Magenta","Magenta","Purple"],
        ["Brown","Red","Purple","Purple"],
        ["Tan","Tan","Light Gray","Light Blue"],
        ["Lime Green","Lime Green","Light Green","Light Blue"],
    ],
    [
        ["Red","Red","Pink","Pink"],
        ["Orange","Light Red","Pink","Pink"],
        ["Orange","Orange","Pink","Pink"],
        ["Yellow","Yellow","Yellow","White"],
    ]
]



colors_after_effects = [
    [
        ["Black", "Deep Blue", "Deep Blue", "Blue"],
        ["Deep Green","Medium Gray-Cyan","Deep Royal Blue","Royal Blue"],
        ["Deep Green","Deep Turquoise","Deep Cyan","Royal Blue"],
        ["Green","Turquoise","Turquoise","Cyan"],
    ],
    [
        ["Deep Red","Deep Magenta","Deep Purple","Purple"],
        ["Deep Yellow","Dark Gray","Medium Gray-Blue","Medium Blue"],
        ["Deep Lime Green","Medium Gray-Green","Medium Gray-Cyan","Medium Royal Blue"],
        ["Lime Green","Medium Green","Medium Turquoise","Medium Cyan"],
    ],
    [
        ["Deep Red","Deep Magenta-Red","Deep Magenta","Purple"],
        ["Deep Orange","Medium Gray-Red","Medium Gray-Magenta","Medium Purple"],
        ["Deep Yellow","Medium Gray-Yellow","Light Gray","Medium Blue"],
        ["Lime Green","Medium Lime Green","Medium Green","Medium Cyan"],
    ],
    [
        ["Red","Magenta-Red","Magenta-Red","Megenta"],
        ["Orange","Medium Red","Medium Magenta-Red","Medium Magenta"],
        ["Orange","Medium Orange","Medium Red","Medium Magenta"],
        ["Yellow","Medium Yellow","Medium Yellow","White"]
    ]
]

for r in range(0,4):
    for g in range(0,4):
        for b in range(0,4):
            af = colors_after_effects[r][g][b]
            caleb = colors_caleb[r][g][b]
            ben = colors_ben[r][g][b]
            first_tab = "\t"
            secont_tab = "\t"
            if len(af) < 8:
                first_tab = "\t\t\t"
            elif len(af) < 16:
                first_tab = "\t\t"
            if len(caleb) < 8:
                second_tab = "\t\t\t"
            elif len(caleb) < 16:
                second_tab = "\t\t"
            print("{}{}|  {}".format(
                # af,
                # first_tab,
                caleb,
                second_tab,
                ben
            ))
