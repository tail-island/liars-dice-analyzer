import json
import numpy as np
import random

from funcy    import *
from pathlib  import Path, PurePath
from operator import eq, getitem


def load_data():
    def load_json(file):
        with open(file, mode='rb') as f:
            return json.load(f)

    def data_set(game_informations):
        def program_paths():
            def all_program_paths():
                for game_information in game_informations:
                    for program in game_information['programs']:
                        yield program['path']

            return tuple(sorted(distinct(all_program_paths())))

        def create_xs(program_path):
            def create_x_parts():
                def all_actions(game):
                    for i in count():
                        for player in game['players']:
                            if i == len(player['actions']):
                                return

                            yield player['actions'][i]

                def bid_face_array(bid_face):
                    result = [0] * 5

                    result[bid_face - 2] = 1

                    return result

                for game_information in game_informations:
                    program = first(filter(lambda program: program['path'] == program_path, game_information['programs']))

                    if not program:
                        continue

                    player = first(filter(lambda player: player['id'] == program['id'], game_information['game']['players']))

                    if not player['actions']:
                        continue

                    face_counts = tuple(map(lambda targetFace: ilen(filter(partial(eq, targetFace), player['faces'])) / 5, range(1, 7)))  # / 5しているのは、範囲を0〜1におさめるため。
                    secret_dice_count = (sum(map(rcompose(rpartial(getitem, 'faces'), len), game_information['game']['players'])) - len(player['faces'])) / 25

                    actions = tuple(all_actions(game_information['game']))
                    player_index = game_information['game']['players'].index(player)

                    for previous_action, action in map(first, chunks(len(game_information['game']['players']), drop(player_index, zip(concat((None,), actions), actions)))):
                        previous_bid = previous_action['bid'] if previous_action else None

                        yield tuple(concat(face_counts,
                                           (secret_dice_count,),
                                           concat(bid_face_array(previous_action['bid']['face']), (previous_action['bid']['min_count'] / 20,)) if previous_action else (0,) * 6,
                                           concat(bid_face_array(action['bid']['face']), (action['bid']['min_count'] / 20,)) if 'bid' in action else (0,) * 6,
                                           (1,) if 'challenge' in action else (0,)))

            return tuple(partition((6 + 1 + 6 + 6 + 1) * 5, cat(create_x_parts())))

        def create_ys(program_path):
            def create_y():
                directory = PurePath(program_path).parts[-2]

                if directory == 'csharp' or directory == 'hardhead' or directory == 'java':
                    return 0

                if directory == 'fool':
                    return 1

                if directory == 'optimist':
                    return 2

                if directory == 'pessimist':
                    return 3

                if directory == 'timid':
                    return 4

            return repeat(create_y())

        def create_xys():
            for program_path in program_paths():
                for x, y in zip(create_xs(program_path), create_ys(program_path)):
                    yield x, y

        xys_collection = tuple(map(tuple, partition_by(second, sorted(create_xys(), key=second))))
        xs, ys = zip(*random.sample(*juxt(identity, len)(tuple(mapcat(rpartial(random.sample, min(map(len, xys_collection))), xys_collection)))))

        return (np.array(xs[1000:]), np.array(ys[1000:])), (np.array(xs[:1000]), np.array(ys[:1000]))

    random.seed(0)

    return data_set(tuple(mapcat(load_json, sorted(Path('./data').glob('*.json')))))
