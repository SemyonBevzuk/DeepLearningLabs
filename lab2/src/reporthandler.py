import os
import json

left_rule = {'<': ':', '^': ':', '>': '-'}
right_rule = {'<': '-', '^': ':', '>': ':'}

def evalute_field(record, field_spec):
    """
    Evalute a field of a record using the type of the field_spec as a guide.
    """
    if type(field_spec) is int:
        return str(record[field_spec])
    elif type(field_spec) is str:
        return str(getattr(record, field_spec))
    else:
        return str(field_spec(record))

def fill_table(records, fields, headings, alignment = None):
    """
    Generate a Doxygen-flavor Markdown table from records.

    file -- Any object with a 'write' method that takes a single string
        parameter.
    records -- Iterable.  Rows will be generated from this.
    fields -- List of fields for each row.  Each entry may be an integer,
        string or a function.  If the entry is an integer, it is assumed to be
        an index of each record.  If the entry is a string, it is assumed to be
        a field of each record.  If the entry is a function, it is called with
        the record and its return value is taken as the value of the field.
    headings -- List of column headings.
    alignment - List of pairs alignment characters.  The first of the pair
        specifies the alignment of the header, (Doxygen won't respect this, but
        it might look good, the second specifies the alignment of the cells in
        the column.

        Possible alignment characters are:
            '<' = Left align (default for cells)
            '>' = Right align
            '^' = Center (default for column headings)
    """

    num_columns = len(fields)
    assert len(headings) == num_columns

    # Compute the table cell data
    columns = [[] for i in range(num_columns)]
    for record in records:
        for i, field in enumerate(fields):
            columns[i].append(evalute_field(record, field))

    # Fill out any missing alignment characters.
    extended_align = alignment if alignment != None else []
    if len(extended_align) > num_columns:
        extended_align = extended_align[0:num_columns]
    elif len(extended_align) < num_columns:
        extended_align += [('^', '<')
                           for i in range[num_columns-len(extended_align)]]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    field_widths = [len(max(column, key=len)) if len(column) > 0 else 0
                    for column in columns]
    heading_widths = [max(len(head), 2) for head in headings]
    column_widths = [max(x) for x in zip(field_widths, heading_widths)]

    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(heading_align, column_widths)])
    heading_template = '| ' + _ + ' |'
    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(cell_align, column_widths)])
    row_template = '| ' + _ + ' |'

    _ = ' | '.join([left_rule[a] + '-'*(w-2) + right_rule[a]
                    for a, w in zip(cell_align, column_widths)])
    ruling = '| ' + _ + ' |'

    table = []
    table.append(heading_template.format(*headings).rstrip() + '\n')
    table.append(ruling.rstrip() + '\n')
    for row in zip(*columns):
        table.append(row_template.format(*row).rstrip() + '\n')

    return table


def generate_table(save_folder_log):
    records = []
    for file in os.listdir(save_folder_log):
        if file.endswith(".json"):
            record = [os.path.splitext(file)[0]]
            with open(os.path.join(save_folder_log, file), 'r') as read_file:
                model_info = json.load(read_file)
            record.append(round(model_info['Statistics']['Test_accuracy'], 4))
            record.append(round(model_info['Statistics']['Test_loss'], 4))
            record.append(round(model_info['Statistics']['Train_accuracy'], 4))
            record.append(round(model_info['Statistics']['Train_loss'], 4))
            record.append(round(model_info['Statistics']['Time_train'], 4))

            records.append(record)
    records.sort(key=lambda x: len(x[0]))

    headings = ['Model name', 'Test accuracy.', 'Test loss', 'Train accuracy', 'Train loss',
                'Time_train (s)']
    fields = [0, 1, 2, 3, 4, 5]
    align = [('^', '<'), ('^', '^'), ('^', '^'), ('^', '^'), ('^', '^'),
             ('^', '^')]

    return fill_table(records, fields, headings, align)


def add_table_to_report(report_path, log_folder):
    table = generate_table(log_folder)

    with open(report_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    table_start_index = 0
    table_end_index = 0
    for i, line in enumerate(lines):
        if line == '[comment]: # (table_start)\n':
            table_start_index = i
        if line == '[comment]: # (table_end)\n':
            table_end_index = i
            break

    for i in range(0, table_end_index - table_start_index - 1):
        del lines[table_start_index + 1]

    new_lines = []
    new_lines.extend(lines[:table_start_index + 1])
    new_lines.extend('\n')
    new_lines.extend(table)
    new_lines.extend('\n')
    new_lines.extend(lines[table_start_index + 1:])

    with open(report_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)
