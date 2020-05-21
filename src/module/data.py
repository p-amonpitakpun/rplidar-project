import csv


def load_to_lists(path=None):
    if path is None:
        raise Exception('No Path Input')
    field_names = []
    stream = []
    with open(path, newline='\r\n') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',')
        field_names = csvreader.fieldnames
        for row in csvreader:
            context = []
            for field_name in field_names:
                try:
                    var = float(row[field_name])
                    context.append(var)
                except ValueError:
                    context.append(row[field_name])
            stream.append(tuple(context))
    return field_names, stream


def load_to_dicts(path=None):
    if path is None:
        raise Exception('No Path Input')
    field_names = []
    stream = []
    with open(path, newline='\r\n') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',')
        field_names = csvreader.fieldnames
        for row in csvreader:
            context = dict()
            for field_name in field_names:
                text = row[field_name]
                try:
                    var = float(text)
                    context[field_name] = var
                except ValueError:
                    context[field_name] = text
            stream.append(context)
    return field_names, stream


if __name__ == '__main__':
    FIELD_NAMES, STREAM = load_to_lists('./data/stream/room.csv')
    print(FIELD_NAMES)
    print(STREAM[:5])
