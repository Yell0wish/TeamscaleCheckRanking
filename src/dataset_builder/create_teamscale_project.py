import json
import time
import requests
import pandas as pd
import subprocess as sp
from datetime import datetime
from pathlib import Path
import re
import logging
import csv
import sys


from ts_client import TS_Client
from teamscale_client.utils import to_dict
from teamscale_client.constants import ConnectorType
from teamscale_client.teamscale_client_config import TeamscaleClientConfig
from teamscale_client.data import GitSourceCodeConnectorConfiguration, ProjectConfiguration

def load_config(config_path: str = "config.json"):
    cfg = {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg

config = load_config()

def create_typeid_severity_dict(csv_file_path):
    """
    读取 CSV 文件并创建一个 typeid: severity 字典。

    :param csv_file_path: CSV 文件路径
    :return: 字典，格式为 {typeid: severity}
    """
    typeid_severity_dict = {}

    # 读取 CSV 文件
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 获取 typeid 和 severity
            typeid = row.get("typeid")
            severity = row.get("severity")
            if typeid and severity:  # 确保字段不为空
                # print(severity[8:])
                # 去掉 severity 值中的 "Default:" 前缀
                typeid_severity_dict[typeid] = severity[8:]

    return typeid_severity_dict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TEAMSCALE_URL = config["Teamscale_url"]
USERNAME = config["Teamscale_username"]
ACCESS_TOKEN = config["Teamscale_access_token"] # 在teamscale客户端上的Access Keys
BASE_PATH = Path(config["Teamscale_working_dir"])
TEAMSCALE_PATH = Path(config["Teamscale_path"])
ANALYSIS_PROFILE_FILE = Path(config["Teamscale_java_profile"])
NEW_REPOS_PATH = Path(config["repos_dir"])
FINDGING_CHURNS_RAW_PATH = Path(config["finding_churns_raw_dir"])
FINDGING_CHURNS_PROCESSED_PATH = Path(config["finding_churns_processed_dir"])
TYPEID_SEVERITY_DICT = create_typeid_severity_dict(Path(config["checks"]))
DATASET_PROJECT_MAP = {
    "ambari": "ambari",
    "amq": "activemq",
    "bookkeeper": "bookkeeper",
    "calcite": "calcite",
    "cassandra": "cassandra",
    "groovy": "groovy",
    "hbase": "hbase",
    "hive": "hive",
    "ignite": "ignite",
    "log4j2": "logging-log4j2",
    "mahout": "mahout",
    "mng": "maven",
    "nifi": "nifi",
    "nutch": "nutch",
    "storm": "storm",
    "tika": "tika",
    "ww": "struts",
    "zookeeper": "zookeeper"
}
INCLUDE_PATTERN = "**.java, **.architecture"
EXCLUDE_PATTERN = "**/package-info.java, **/module-info.java"
ANALYSIS_PROFILE = "Java (default)"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 读取 CSV 文件并生成字典
def load_commit_dict(file_path):
    """
    从 CSV 文件中读取 File Name 和 Commit Hash 并返回一个字典
    """
    try:
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 将 'File Name' 列作为键，'Commit Hash' 列作为值，生成字典
        commit_dict = pd.Series(df['Commit Hash'].values, index=df['File Name']).to_dict()
        
        return commit_dict
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径！")
        return {}
    except KeyError:
        print("文件中没有找到 'File Name' 或 'Commit Hash' 列，请确认文件格式！")
        return {}

def init_TS_Client() -> TS_Client:
    # Using TSconfig to avoid using dummyProjectId when no projects exist yet
    TSconfig = TeamscaleClientConfig(
        TEAMSCALE_URL, USERNAME, ACCESS_TOKEN)
    # Notice that an extented TeamscaleClient is used
    client = TS_Client.from_client_config(TSconfig)

    client.internal_post_import_analysis_profile(
        {'analysis-profile': ANALYSIS_PROFILE_FILE.open('rb'), })
    
    client.internal_put_options_server(
        'file-system-access', {'accessibleFilepaths': str(BASE_PATH)})
    
    return client

def start_server():
    """启动 Teamscale 服务器"""
    server_process = sp.Popen(['cmd', '/c', 'start', 'teamscale.bat'], cwd=str(TEAMSCALE_PATH))

    # 轮询等待服务器启动
    max_retries = 60  # 最多尝试 60 次
    for _ in range(max_retries):
        try:
            response = requests.get(TEAMSCALE_URL, timeout=10, proxies={"http": None, "https": None}) # type: ignore
            if response.status_code == 200:
                print("Teamscale server is up!")
                return server_process
            else :
                print('code: ',response.status_code)
        except requests.ConnectionError:
            pass
        time.sleep(10)  # 每次等待 10 秒

    raise RuntimeError("Teamscale server failed to start within the expected time.")

def stop_server(server_process):
    """关闭 Teamscale 服务器"""
    server_process.terminate()  # 关闭进程
    try:
        sp.run(['taskkill', '/F', '/IM', 'java.exe'], check=True)
        print(f"Successfully terminated process: {'java.exe'}")
    except sp.CalledProcessError as e:
        print(f"Failed to terminate process {'java.exe'}: {e}")
    time.sleep(10)  # 等待服务器完全关闭

def create_teamscale_project(client : TS_Client, repoName, branch, start_date, repo_path=None):
    assert repo_path is not None
    repo_path = f"file://{repo_path.as_posix()}"
    print(repo_path)

    existing_projects, waitTime = [], 0
    for p in to_dict(client.get_projects()):
        existing_projects.extend(p['publicIds'])

    project_id = repoName + '_id'
    if project_id not in existing_projects:

        # repo_path = f"file://{REPOS_PATH / repoName}.git"

        # repo_path = f"file://F:\\FilesForResearch\\codeqlResearchProject\\CheckMatch\\new_repo"
        

        account_name = 'Account_TD_Analyzer'

        data = {"connectorTypeInfo": {"connectorEnum": ConnectorType.GIT,
                                        "connectorType": "SOURCE_CODE_REPOSITORY"},
                "credentialsName": account_name, "password": "admin_repo",
                "uri": repo_path, "username": "admin_repo"}
        client.internal_post_add_external_account(data)

        connector_config = GitSourceCodeConnectorConfiguration(
            repository_identifier=(
                'Repo_' + repoName),
            default_branch_name=branch,
            account=account_name,
            start_revision='', end_revision='',
            included_file_names=INCLUDE_PATTERN,
            excluded_file_names=EXCLUDE_PATTERN,
            enable_branch_analysis=True,
            preserve_empty_commits=True
        )

        project_configuration = ProjectConfiguration(name=repoName, project_id=project_id,
                                                        profile=ANALYSIS_PROFILE, connectors=[connector_config])
        p_config = to_dict(project_configuration)
        p_config["branchingConfiguration"] = {"branchConfigurations": [
            {"branchNamePattern": "main|master|trunk|release.*",
                "startDate": start_date},
            {"branchNamePattern": ".*", "startDate": start_date}]}
        project_configuration = p_config
        # print(json.dumps(project_configuration, indent=4))
        client.create_project(project_configuration) # type: ignore

        # Giving Teamscale enough time to create the project before trying to query for that project_id
        waitTime = 60
        print(f'Created project with id: {project_id}')

    return waitTime

def check_projects_initialization(client : TS_Client , project_id, waitTime, maxRepeat=1440):
    print(
        f'Starting to wait for Teamscale to finish its internal analysis of {project_id}.')
    time.sleep(waitTime)
    ts_analysis_complete = False
    while maxRepeat > 0:
        maxRepeat -= 1
        res = client.internal_get_branch_analysis_state(
            project_id, ' ')
        ts_analysis_complete = True if (
            res["state"] == 'LIVE_ANALYSIS') else False
        if ts_analysis_complete:
            maxRepeat = 0
        else:
            time.sleep(60)

    # Double check
    assert (ts_analysis_complete and ('Status summary: OK (RC: 0)' in client.get_health_check(
    ))), f'Error while waiting for Teamscale initialization to finish'

    print(f'Teamscale initialization for {project_id} finished successfully.')

def delete_teamscale_project(client : TS_Client, repo_arg):
    """Deletes one or all projects in Teamscale

    Args:
        del_arg: specified via a flag when starting td_analyzer.py
    """
    client.delete_project((repo_arg + '_id'))
    print(f'Deleted Teamscale project: {repo_arg}')

def get_repo_df(client : TS_Client, repoName):
    repo_df = {'branch': [], 'commitHash': [], 'parentCommitsCount': [], 'timeStamp': [],
                'authorName': [], 'authorEmail': [], 'prevTimeStamp': [], 'prevBranch': []}
    repolog = client.internal_get_repo_log(repoName + '_id')
    # self.write_repologs(repoName, repolog)
    for e in repolog:
        repo_df['branch'].append(e['logEntry']['commit']['branchName'])
        repo_df['commitHash'].append(e['logEntry']['revision'])
        repo_df['timeStamp'].append(e['logEntry']['commit']['timestamp'])
        repo_df['authorName'].append(e['logEntry']['author'])

        cnt_pc = len(e['logEntry']['commit']['parentCommits'])
        repo_df['parentCommitsCount'].append(cnt_pc)

        if cnt_pc > 0:
            repo_df['authorEmail'].append(e['logEntry']['mail'])
            parent = e['logEntry']['commit']['parentCommits'][0]
            repo_df['prevTimeStamp'].append(parent['timestamp'])
            repo_df['prevBranch'].append(parent['branchName'])
        else:
            repo_df['authorEmail'].append('')
            repo_df['prevTimeStamp'].append(e['logEntry']['commit']['timestamp'])
            repo_df['prevBranch'].append(e['logEntry']['commit']['branchName'])

    return pd.DataFrame(repo_df)

def get_findings(client : TS_Client, repo_df, project_id, name, version):
    assert len(repo_df.index) == 1
    idx2 = 0
    t1 = str(repo_df['timeStamp'][idx2])
    b1 = repo_df['branch'][idx2]
    commit_hash = repo_df['commitHash'][idx2]


    find_churn = client.get_findings_churn_list(project_id, t1, b1)

    # 以下是内容检查
    added_findings = find_churn.get("addedFindings", [])

    output_csv_file = FINDGING_CHURNS_PROCESSED_PATH / f"{name}-{version}.csv"
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 写入标题行
        writer.writerow(["type", "severity", "group", "category", "start_line", "end_line", "location"])

        for finding in added_findings:
            group_name = finding.get("groupName", "")
            category_name = finding.get("categoryName", "")
            message = finding.get("message", "")
            raw_start_line = finding.get("location", {}).get("rawStartLine", "")
            raw_end_line = finding.get("location", {}).get("rawEndLine", "")
            location = finding.get("location", {}).get("location", "")
            assessment = finding.get("assessment", "")
            type_id = finding.get("typeId", "")

            type_checking = type_id.split('/')[-1]
            if type_checking in TYPEID_SEVERITY_DICT:
                if assessment.lower() == TYPEID_SEVERITY_DICT[type_checking].lower():
                    writer.writerow([type_checking, assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                else:
                    raise ValueError(f"In {name} {version}, the severity in chrun is {assessment}, but in findings.csv is {TYPEID_SEVERITY_DICT[type_checking]}")
            else:
                if type_id == 'Naming/JAVA' and assessment.lower() == 'yellow':
                    # 对应的type是Java naming conventions
                    writer.writerow(['Java naming conventions', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Comments/Missing Interface Comment' and assessment.lower() == 'yellow':
                    # 对应的type是Interface comment completeness
                    writer.writerow(['Interface comment completeness', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Metric Violations/LSL' and group_name == 'Method Length':
                    # 对应的type是Long Method
                    # severity是auto
                    writer.writerow(['Long Method', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Comments/Unrelated Member Comment' and assessment.lower() == 'yellow':
                    # 对应的type是Unrelated interface comment
                    writer.writerow(['Unrelated interface comment', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Comments/Task Tags' and assessment.lower() == 'yellow':
                    # 对应的type是Task tags
                    writer.writerow(['Task tags', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Metric Violations/Nesting Depth':
                    # 对应的type是Deep Nesting
                    # severity是auto
                    writer.writerow(['Deep Nesting', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Metric Violations/SLOC':
                    # 对应的type是Long File
                    # severity是auto
                    writer.writerow(['Long File', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Comments/Commented Out Code' and assessment.lower() == 'yellow':
                    # 对应的type是Commented-out code
                    writer.writerow(['Commented-out code', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id == 'Comments/Problem Tags' and assessment.lower() == 'red':
                    # 对应的type是Problem tags
                    writer.writerow(['Problem tags', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                elif type_id =='Comments/Empty Interface Comment' and assessment.lower() == 'yellow':
                    # 对应的type是Empty interface comment
                    writer.writerow(['Empty interface comment', assessment, group_name, category_name, raw_start_line, raw_end_line, location])
                else:
                    print(message)
                    raise ValueError(f"Typeid {type_id} with group {group_name} and category {category_name} not found in findings.csv")

    # 以下是写入原始数据
    # 构建输出文件名（根据 commit hash 命名）   
    output_file = FINDGING_CHURNS_RAW_PATH / f"{name}-{version}.json" # 需要修改

    # 将数据写入 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(find_churn, f, ensure_ascii=False, indent=4)

    # print(f"Saved findings churn to {output_file}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # 终端输出
            logging.FileHandler(filename='create_teamscale_project.log', mode="a", encoding="utf-8")  # 写入日志文件
        ]
    )

    # 确保 stdout 也是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8") # type: ignore

    FINDGING_CHURNS_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    FINDGING_CHURNS_RAW_PATH.mkdir(parents=True, exist_ok=True)


    server_process = start_server()
    client = init_TS_Client()

    commit_dict = load_commit_dict(Path(config["successful_projects"]))

    for index, file_name in enumerate(commit_dict.keys()):
        pattern = r"(\w+)-([\d\.]+)"
        match = re.search(pattern, file_name)
        assert match is not None
        name = match.group(1)  # 形如"ambari"
        version = match.group(2)  # 形如"1.2.0"
        
        # 创建 Teamscale 项目
        logging.info(f"Creating Teamscale project for {name} {version}")
        start_time = time.time()
        waitTime = create_teamscale_project(client, f"{name}-{version.replace('.', '_')}", 'master', start_date=datetime.utcfromtimestamp(
                    0).strftime('%Y-%m-%d'), repo_path=NEW_REPOS_PATH / f'{name}-{version}')
        check_projects_initialization(client, f"{name}-{version.replace('.', '_')}" + '_id', waitTime)
        print(f'Successfully created Teamscale project for {name} {version}')
        logging.info(f'Successfully created Teamscale project for {name} {version}')
        logging.info(f"Time taken to create Teamscale project: {time.time() - start_time}")
        # if index != 0 and index % 50 == 0:
        #     stop_server(server_process)
        #     time.sleep(20)
        #     server_process = start_server()
        #     client = init_TS_Client()

        # 收集teamscale分析信息
        logging.info(f"Collecting findings churn for {name} {version}")
        repo_df = get_repo_df(client, f"{name}-{version.replace('.', '_')}")
        get_findings(client, repo_df, f"{name}-{version.replace('.', '_')}" + '_id', name, version)
        logging.info(f"Successfully collected findings churn for {name} {version}")

    stop_server(server_process)

if __name__ == "__main__":
    main()