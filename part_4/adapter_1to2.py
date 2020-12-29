class Adapter_1to2:

    def filter_same_candidates(self, all_candidates):
        red_x, red_y, green_x, green_y = all_candidates[0], all_candidates[1], all_candidates[2], all_candidates[3]
        for r_x, r_y in zip(red_x, red_y):
            for g_x, g_y in zip(green_x, green_y):
                if r_x == g_x and r_y == g_y:
                    green_x -= g_x
                    green_y -= g_y

        return red_x, red_y, green_x, green_y

    def adapt(self, all_candidates):
        red_x, red_y, green_x, green_y = self.filter_same_candidates(all_candidates)
        candidates = []
        auxiliary = []
        for x, y in zip(red_x, red_y):
            candidates.append((x, y))
            auxiliary.append("R")
        for x, y in zip(green_x, green_y):
            candidates.append((x, y))
            auxiliary.append("G")

        return candidates, auxiliary

