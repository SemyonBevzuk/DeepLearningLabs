import os
import json

left_rule = {'<': ':', '^': ':', '>': '-'}
right_rule = {'<': '-', '^': ':', '>': ':'}

def evalute_field(record, field_spec):
    if type(field_spec) is int:
        return str(record[field_spec])
    elif type(field_spec) is str:
        return str(getattr(record, field_spec))
    else:
        return str(field_spec(record))

def fill_table(records, fields, headings, alignment = None):
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


def find_old_table(lines, open_tag, close_tag):
    table_start_index = 0
    table_end_index = 0
    for i, line in enumerate(lines):
        if line == open_tag:
            table_start_index = i
        if line == close_tag:
            table_end_index = i
            break
    return (table_start_index, table_end_index)


def delete_old_table(lines, table_start_index, table_end_index):
    for i in range(0, table_end_index - table_start_index - 1):
        del lines[table_start_index + 1]


def create_lines_with_new_table(lines, table, input_index):
    new_lines = []
    new_lines.extend(lines[:input_index + 1])
    new_lines.extend('\n')
    new_lines.extend(table)
    new_lines.extend('\n')
    new_lines.extend(lines[input_index + 1:])
    return new_lines

def generate_result_table(save_folder_log):
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

    headings = ['Model name', 'Test accuracy', 'Test loss', 'Train accuracy', 'Train loss',
                'Time_train (s)']
    fields = [0, 1, 2, 3, 4, 5]
    align = [('^', '<'), ('^', '^'), ('^', '^'), ('^', '^'), ('^', '^'),
             ('^', '^')]

    return fill_table(records, fields, headings, align)


def generate_graph_table(img_folder):
    files = []
    for file in os.listdir(img_folder):
        if file.endswith(".png"):
            files.append(file)
    files.sort(reverse=True)

    records = []
    for loss, accuracy in zip(files[0::2], files[1::2]):
        records.append(['![](img/' + accuracy + ')', '![](img/' + loss + ')'])

    headings = ['Accuracy', 'Loss']
    fields = [0, 1]
    align = [('^', '<'), ('^', '<')]
    return fill_table(records, fields, headings, align)


def add_result_table_to_report(report_path, img_folder):
    table = generate_result_table(img_folder)

    with open(report_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    open_tag = '[comment]: # (result_table_start)\n'
    close_tag = '[comment]: # (result_table_ens\n'
    table_start_index, table_end_index = find_old_table(lines, open_tag, close_tag)
    delete_old_table(lines, table_start_index, table_end_index)

    new_lines = create_lines_with_new_table(lines, table, table_start_index)

    with open(report_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)


def add_graph_table_to_report(report_path, img_folder):
    table = generate_graph_table(img_folder)

    with open(report_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    open_tag = '[comment]: # (graph_table_start)\n'
    close_tag = '[comment]: # (graph_table_end)\n'
    table_start_index, table_end_index = find_old_table(lines, open_tag, close_tag)
    delete_old_table(lines, table_start_index, table_end_index)

    new_lines = create_lines_with_new_table(lines, table, table_start_index)

    with open(report_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

