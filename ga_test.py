#使用遗传算法输出一个较优课表
#首先要读取CSV文件，包括原生课程表和老师志愿表。
# 初始化种群：
# 可以基于初始课表和老师的志愿信息生成初始种群。种群中的每个个体是对初始课表的一种修改方案，例如，在满足基本排课约束的前提下，对部分课程的时间、教室或教师进行调整。
# 适应度评估（评分函数）：
# 对于每个个体，根据老师的志愿信息进行评分。评分可以基于个体与老师志愿的匹配程度，满足第一志愿的部分给予较高分数，第二志愿次之，第三志愿分数较低。同时，需要考虑基本的排课约束，如时间冲突、教室冲突等。
# 选择操作：
# 从当前种群中选择部分个体作为父代，可使用轮盘赌选择、锦标赛选择等方法，使适应度高的个体有更高概率被选中。
# 交叉操作：
# 对选中的父代个体进行交叉操作，生成子代个体。在交叉时，要确保生成的子代个体仍然满足排课约束。
# 变异操作：
# 以一定概率对新生成的子代个体进行变异，引入新的修改方案，避免陷入局部最优。
# 更新种群：
# 用子代个体替换部分或全部父代个体，形成新的种群。
# 终止条件：
# 设定迭代次数或当种群的平均适应度达到一定水平或最优个体的适应度不再有显著提升时终止算法。
import pandas as pd
import numpy as np
import random
import logging
import pandas as pd
import numpy as np
import random

logger =logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def read_schedule(schedule_file):
    """
    读取初始课表文件
    :param schedule_file: 课表文件路径
    :return: 课表的 DataFrame
    """
    return pd.read_csv(schedule_file,encoding='ANSI',dtype='str')


def read_priority(priority_file):
    """
    读取老师的志愿文件
    :param priority_file: 志愿文件路径
    :return: 志愿的 DataFrame
    """
    return pd.read_csv(priority_file,encoding='ANSI',dtype='str')


def fitness(new_schedule_df, priority_dfs):
    """
    计算适应度函数
    :param new_schedule_df: 新的课表的 DataFrame
    :param priority_dfs: 老师志愿的 DataFrame 列表
    :return: 适应度得分
    """
    score = 0
    for df in priority_dfs:
        for _, row in df.iterrows():
            # 获取志愿信息
            priority = (row['Day'], row['Course_Sequence'],str(row['Application_Priority']),row['Teacher_ID'])
            # 检查新课表中是否采纳了老师的志愿
            if is_priority_adopted(new_schedule_df, priority):
                logger.info(f"Teacher {row['Teacher_ID']} adopts priority {int(row['Application_Priority'])}  : course {row['Course_ID']} on day {row['Day']}")
                if int(row['Application_Priority']) == 1:
                    score += 90
                elif int(row['Application_Priority']) == 2:
                    score += 60
                elif int(row['Application_Priority']) == 3:
                    score += 30
                break
    return score


def is_priority_adopted(schedule_df, priority):
    """
    检查老师的志愿是否被采纳
    :param schedule_df: 课表的 DataFrame
    :param priority: 志愿，包含课程的时间和位置信息，这里是 (Day, Course_Sequence) 元组
    :param week: 老师志愿的周次信息，例如 "1,2,3,4"
    :return: True 采纳，False 未采纳
    """
    day, course_sequence, priority_sequence,teacher_id = priority
    #拼接带优先级的课程ID
    teacher_id_with_priority = teacher_id+'('+priority_sequence+')'
    if day in schedule_df.columns:
        for course in course_sequence.split(','):
            #如果该出现的块未出现，则说明志愿未被采纳
            if teacher_id_with_priority not in (schedule_df.iloc[int(course)-1, int(day)+7]).split(','):
                return False
    return True


def check_conflict(existing_week, new_week):
    """
    检查周次和课程表是否冲突
    :param existing_week: 已有的周次信息，例如 "1,2,3,4"
    :param new_week: 新的周次信息，例如 "5,6,7,8"
    :return: True 有冲突，False 无冲突
    """
    existing_week_list = [x for x in existing_week.split(', ')]
    new_week_list = [x for x in new_week.split(', ')]
    for week in new_week_list:
        if week in existing_week_list:
            return True
    return False


def check_priority(new_schedule_df, priority, row,teacher_id):
    """
    检查志愿是否可以添加到课表中，不能覆盖已有数据
    :param new_schedule_df: 新的课表的 DataFrame
    :param priority: 志愿，包含课程的时间和位置信息，这里是 (Day, Course_Sequence) 元组
    :param teacher_id: 老师的编号，用于检查冲突时避免同一老师在同一时间上课
    :return: True 可以添加，False 不能添加
    """
    # 先检查老师是否已经排过
    # 遍历每一列，检查是否有相同的老师
    for column in range(8, 15):
        for r in range(len(new_schedule_df)):
            if new_schedule_df.iloc[r, column] == '':
                continue
            for ids in new_schedule_df.iloc[r, column].split(','):
                if ids.split('(')[0] == str(teacher_id):
                    return False
    
    # 再检查周次是否有冲突
    day, course_sequence,pr_index = priority
    # 确保 day 在列中
    if str(day) in new_schedule_df.columns:
        # 假设 course_sequence 是多个元素，如 "1,2,3"
        course_sequence_list = [int(c) for c in course_sequence.split(', ')]
        for course in course_sequence_list:
            # 确保 course 在行中
            if course-1 in new_schedule_df.index:
                existing_week = str(new_schedule_df.iloc[course-1, int(day)])
                if existing_week == '0' :
                    return True
                if check_conflict(existing_week, str(row['Week'].iloc[pr_index])):
                    return False
    return False


def mutate(initial_schedule_df, priority_dfs,mutate_rate=5):
    """
    变异操作，尝试将一个老师的志愿添加到课表中
    :param initial_schedule_df: 初始课表的 DataFrame
    :param priority_dfs: 老师志愿的 DataFrame 列表
    :return: 变异后的课表的 DataFrame
    """
    logger.info("Mutating Once...")
    # 初始化新课表
    new_schedule = initial_schedule_df.copy()
    # 随机添加mutate_rate个志愿，可根据实际情况调整
    added_teachers = set()  # 用于记录已经添加过志愿的老师，避免重复添加
    for _ in range(mutate_rate):
        teacher_df = random.choice(priority_dfs)
        teacher_id = int(teacher_df['Teacher_ID'].iloc[0])
        if teacher_df['Week'].iloc[0] == "nan":  # 跳过没有周次的志愿
            added_teachers.add(teacher_id)
            continue
        if teacher_id not in added_teachers:  # 确保不重复添加同一老师的志愿
            first_priority = (teacher_df['Day'].iloc[0], teacher_df['Course_Sequence'].iloc[0],int(teacher_df['Application_Priority'].iloc[0])-1)
            second_priority = (teacher_df['Day'].iloc[1], teacher_df['Course_Sequence'].iloc[1],int(teacher_df['Application_Priority'].iloc[1])-1)
            third_priority = (teacher_df['Day'].iloc[2], teacher_df['Course_Sequence'].iloc[2],int(teacher_df['Application_Priority'].iloc[2])-1)
            match_priority= ()
            if check_priority(new_schedule, first_priority, teacher_df,teacher_id):
                match_priority = first_priority
            elif check_priority(new_schedule, second_priority, teacher_df,teacher_id):
                match_priority = second_priority
            elif check_priority(new_schedule, third_priority, teacher_df,teacher_id):
                match_priority = third_priority
            if match_priority:
                index = int(match_priority[2])
                logger.info("Priority matched!: Teacher_ID:%d,priority:%s" % (teacher_id,index+1))
                for course_seq in match_priority[1].split(', '):
                    origin_weeks= str(new_schedule.iloc[int(course_seq)-1, int(match_priority[0])])
                    if origin_weeks == '0':
                        new_schedule.iloc[int(course_seq)-1, int(match_priority[0])] = teacher_df['Week'].iloc[index]
                        new_schedule.iloc[int(course_seq)-1, int(match_priority[0])+7] = str(teacher_id)+'('+str(index+1)+')'  # 保存课程 ID
                    else:
                        new_content = new_schedule.iloc[int(course_seq)-1, int(match_priority[0])]+', '+teacher_df['Week'].iloc[index]
                        # 对修改后的周次重新按升序排序
                        weeks = list(set(int(x) for x in new_content.split(', ') if x.isdigit()))
                        weeks_str = ', '.join(str(x) for x in sorted(weeks))
                        new_schedule.iloc[int(course_seq)-1, int(match_priority[0])] =weeks_str
                        # 保存课程 ID
                        if new_schedule.iloc[int(course_seq)-1, int(match_priority[0])+7] != '':
                            new_schedule.iloc[int(course_seq)-1, int(match_priority[0])+7] = new_schedule.iloc[int(course_seq)-1, int(match_priority[0])+7]+','+(str(teacher_id)+'('+str(index+1)+')')
                        else:
                            new_schedule.iloc[int(course_seq)-1, int(match_priority[0])+7] = str(teacher_id)+'('+str(index+1)+')'
            added_teachers.add(teacher_id)
    return new_schedule

def check_same_course(parent, object_parent, column):
    """
    检查要替换的那一列的课表是否在父代中有相同的实验课
    :param parent1: 父代课表的 DataFrame 1
    :param parent2: 父代课表的 DataFrame 2
    :return: 子代课表的 DataFrame
    """
    # 替换列的课程信息块集合
    courses_in_column = set()
    course_id_contents = set(object_parent.iloc[:, column+7])
    course_id_contents.discard('')
    for content in course_id_contents:
        courses = content.split(',')
        for course in courses:
            if course.split('(')[0] not in courses_in_column:
                courses_in_column.add(course.split('(')[0])
    #父代课表的课程信息块集合(除去要替换的那一列)
    courses_in_parent = set()
    course_id_contents_in_parent=set() 
    # 遍历每一格 (性能灾难)
    for column in range(8, 15):
        for r in range(len(parent)): 
            if parent.iloc[r, column] not in course_id_contents_in_parent:
                course_id_contents_in_parent.add(parent.iloc[r, column])
                courses = parent.iloc[r, column].split(',')
                for course in courses:
                    if course.split('(')[0] not in courses_in_parent:
                        courses_in_parent.add(course.split('(')[0])
    courses_in_parent.discard('')
    # 两个集合相交，说明有相同的课程 ID
    if courses_in_parent.intersection(courses_in_column):
        return True
    return False
    
        

def crossover(parent1, parent2):
    """
    交叉操作，合并两个课表的部分信息
    :param parent1: 父代课表的 DataFrame 1
    :param parent2: 父代课表的 DataFrame 2
    :return: 子代课表的 DataFrame
    """
    child = parent1.copy()
    child.to_csv('./child_pre.csv', index=False)
    # 随机选择交叉的位置，这里简单地选择一列位置
    rand_choices = list(range(1, 8))
    random_column = random.choice(rand_choices)
    #检查是否有同个老师，避免一个老师多个志愿被排
    while check_same_course(parent1, parent2, random_column):
        rand_choices.remove(random_column)
        if len(rand_choices) == 0:
            # 如果没有合法的列可以交叉。则直接返回父代课表
            return child
        random_column = random.choice(rand_choices)
    # 替换该列信息
    child.iloc[:, random_column] = parent2.iloc[:, random_column]
    child.iloc[:, random_column+7] = parent2.iloc[:, random_column+7]
    child.to_csv('./child_post.csv', index=False)
    return child

def drop_duplicates(schedule_df):
    """
    去重操作，删除重复的课程
    :param schedule_df: 课表的 DataFrame
    :return: 去重后的课表的 DataFrame
    """
    # 遍历每一列，检查是否有重复的课程
    for column in range(8, 15):
        for row in range(len(schedule_df)):
            if schedule_df.iloc[row, column] == '':
                continue
            schedule_df.iloc[row, column] = ','.join(set(schedule_df.iloc[row, column].split(',')))
    return schedule_df


def generate_initial_population(initial_schedule_df, priority_dfs, population_size):
    """
    生成初始种群，先变异一次
    :param initial_schedule_df: 初始课表的 DataFrame
    :param priority_dfs: 老师志愿的 DataFrame 列表
    :param population_size: 种群大小
    :return: 初始种群的列表，包含多个课表的 DataFrame
    """
    population = []
    for _ in range(population_size):
        new_schedule = mutate(initial_schedule_df.copy(), priority_dfs,17)  
        population.append(drop_duplicates(new_schedule))
    return population


def genetic_algorithm(initial_schedule_df, priority_dfs, population_size=10, generations=100):
    """
    遗传算法的主函数
    :param initial_schedule_df: 初始课表的 DataFrame
    :param priority_dfs: 老师志愿的 DataFrame 列表
    :param population_size: 种群大小
    :param generations: 迭代次数
    :return: 最优课表的 DataFrame
    """
    # 生成初始种群
    logger.info("Generating initial population...")
    population = (generate_initial_population(initial_schedule_df, priority_dfs, population_size))
    for gen in range(generations):
        new_population = []
        # 计算适应度
        logger.info("Generation %d start! " % (gen+1))
        fitness_scores = [fitness(df, priority_dfs) for df in population]
        best_index = np.argmax(fitness_scores)
        best_schedule = population[best_index]
        new_population.append(best_schedule)
        while len(new_population) < population_size:
            # 选择
            parent1 = random.choices(population, weights=fitness_scores)[0]
            parent2 = random.choices(population, weights=fitness_scores)[0]
            # 交叉
            logger.info("Crossover Once...")
            child = crossover(parent1, parent2)
            # 变异
            child = mutate(child, priority_dfs)
            # 去重
            child = drop_duplicates(child)
            new_population.append(child)
        population = new_population
        logger.info("Generation %d done! " % (gen+1))
        logger.info("Generation %d: Best fitness score = %d" % (gen+1, max(fitness_scores)))

    return best_schedule


if __name__ == '__main__':
    schedule_file = './data/schedule_for_teachers/1.CSV'
    initial_schedule_df = read_schedule(schedule_file)
    # 为课表添加额外的列，用于存储课程 ID
    days = initial_schedule_df.columns[1:]
    for day in days:
        initial_schedule_df[day + '_Priority_'] = ''
    priority_dfs = []
    for i in range(1, 17):
        priority_file = './data/priorities_for_teachers/{0}p.CSV'.format(i)
        priority_dfs.append(read_priority(priority_file))
    final_schedule = genetic_algorithm(initial_schedule_df, priority_dfs)
    # 保存最终课表为 CSV 文件
    final_schedule.to_csv('./final_schedule.csv', index=False)