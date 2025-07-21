import re

with open("table.tex", "r") as file:
    lines = file.readlines()

rows = []
for line in lines:
    if '&' in line:
        parts = [part.strip() for part in line.strip().rstrip(r'\\').split('&')]
        rows.append(parts)
    else:
        rows.append(line.strip())

output_lines = []
i = 0
while i < len(rows):
    if isinstance(rows[i], list) and len(rows[i]) > 1 and rows[i][1] == 'p-value':
        p_row = rows[i]
        rho_row = rows[i + 1]

        new_p = p_row[:2]
        new_rho = rho_row[:2]

        for pval, rho in zip(p_row[2:], rho_row[2:]):
            try:
                pval_clean = pval.replace(r'\textless{}', '').replace('<', '').strip()
                pval_num = float(pval_clean)

                if pval_num < 0.05:
                    rho=float(rho)
                    if rho >= 0.49:
                        rho= rf'\cellcolor[rgb]{{0.0, 0.4, 0.0}}\textcolor{{white}}{{{rho}}}'
                    elif rho >= 0.29:
                        rho= rf'\cellcolor[rgb]{{0.0, 0.75, 0.0}}{rho}'
                    elif rho >= 0.1:
                        rho= rf'\cellcolor[rgb]{{0.1, 0.9, 0.1}}{rho}'
                    else:
                        rho= rf'\cellcolor[rgb]{{0.65, 1.0, 0.65}}{rho}'
                    pval = rf'\cellcolor{{lightgray}}{pval}'
                    #rho = rf'\cellcolor{{lightgray}}{rho}'

                    

            except ValueError:
                # Non-numeric or missing p-value (e.g., '-')
                pass

            new_p.append(pval)
            new_rho.append(rho)

        output_lines.append(' & '.join(new_p) + r' \\')
        output_lines.append(' & '.join(new_rho) + r' \\')
        i += 2
    elif isinstance(rows[i], list):
        output_lines.append(' & '.join(rows[i]) + r' \\')
        i += 1
    else:
        output_lines.append(rows[i])
        i += 1

with open("highlighted_table.tex", "w") as f:
    for line in output_lines:
        f.write(line + '\n')
