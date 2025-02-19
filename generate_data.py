import pandas as pd
import numpy as np


# 生成初始课表
def generate_initial_schedule():
    days = ['1', '2', '3', '4', '5', '6', '7']
    courses = list(range(1, 13))
    schedule = pd.DataFrame('0', index=courses, columns=days)

    num_existing_courses = np.random.randint(10, 30)
    for _ in range(num_existing_courses):
        day = np.random.choice(days)
        course = np.random.choice(courses)
        total_weeks = np.random.choice([2, 4, 8])
        start_week = np.random.randint(1, 17 - total_weeks + 1)
        end_week = start_week + total_weeks - 1
        week_list = list(range(start_week, end_week + 1))
        week_str = ','.join(str(w) for w in week_list)
        schedule.at[course, day] = week_str

    return schedule


# 生成单个老师的志愿表（包含三个志愿）
def generate_teacher_priority(teacher_id):
    days = ['1', '2', '3', '4', '5', '6', '7']
    courses = list(range(1, 13))
    data = []
    for _ in range(3):
        day = np.random.choice(days)
        course_sequence = np.random.choice(courses)
        total_weeks = np.random.choice([2, 4, 8])
        start_week = np.random.randint(1, 17 - total_weeks + 1)
        end_week = start_week + total_weeks - 1
        week_list = list(range(start_week, end_week + 1))
        week_str = ','.join(str(w) for w in week_list)
        priority = _ + 1
        data.append({
            'Teacher_ID': teacher_id,
            'Day': day,
            'Course_Sequence': course_sequence,
            'Course_ID': 'A'+str(teacher_id),
            'Week': week_str,
            'Application_Priority': priority
        })
    return pd.DataFrame(data)


# 生成一组数据（一个课表和 10 个老师的志愿表）
def generate_data_set(set_index):
    initial_schedule = generate_initial_schedule()
    initial_schedule.to_csv(f'./mnt/initial_schedule_set_{set_index}.CSV', index_label='Course_Sequence')

    for teacher_id in range(1, 11):
        teacher_priority = generate_teacher_priority(teacher_id)
        teacher_priority.to_csv(f'./mnt/schedule_{teacher_id}_priority_set_{set_index}.CSV', index=False)


# 生成 10 组数据
for set_index in range(1, 11):
    generate_data_set(set_index)