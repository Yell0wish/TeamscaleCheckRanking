from typing import Optional

from teamscale_client import TeamscaleClient
from teamscale_client.utils import to_json
from teamscale_client.teamscale_client_config import TeamscaleClientConfig


class TS_Client(TeamscaleClient):
    """Extends the TeamscaleClient with not implemented API functions
    """
    @staticmethod
    def from_client_config(
            config: TeamscaleClientConfig, sslverify: bool = True, timeout: float = 60.0, branch: Optional[str] = None) -> 'TS_Client':
        """Overriding
        """
        return TS_Client(config.url, config.username, config.access_token, config.project_id, sslverify, timeout,
                         branch)

    # ---------- INTERNAL API ----------

    def internal_put_options_server(self, option_id, data):
        """
        """
        return self.put(
            f"{self._api_url}/options/server/{option_id}?save-if-validation-fails=false",
            data=to_json(data),
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )

    def internal_post_add_external_account(self, data):
        """
        Adds local git account to Teamscale. Used for connecting 
        data=
        {
            "connectorTypeInfo": {
                "connectorEnum": "string",
                "connectorType": "SOURCE_CODE_REPOSITORY"
            },
            "credentialsName": "string",
            "password": "string",
            "uri": "string",
            "username": "string"
        }
        """
        return self.post(
            f"{self._api_url}/external-accounts",
            headers={'Accept': '*/*',
                     'Content-Type': 'application/json'},
            json=data
        )

    def internal_post_import_analysis_profile(self, file_data):
        """
        type(file_data) = dict
        """
        return self.post(
            f"{self._api_url}/analysis-profiles/import",
            headers={'Accept': '*/*'},
            files=file_data
        )

    def internal_get_metrics_delta(self, project_id, t1, branch1, t2, branch2, max_ms=-1, uniform_path=''):
        """
        """
        response = self.get(
            f"{self._api_url}/projects/{project_id}/metrics/delta",
            params={
                "t1": f'{branch1}:{t1}',
                "t2": f'{branch2}:{t2}',
                "max-milliseconds": max_ms,
                "uniform-path": uniform_path,
            },
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def internal_get_project_postponed_rollbacks(self, project_id):
        """
        Rollbacks for a given project
        """
        response = self.get(
            f"{self._api_url}/projects/{project_id}/postponed-rollbacks",
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def internal_get_repo_log(self, project_id):
        """Returns all to Teamscale known commits for given project_id
        """
        response = self.get(
            f"{self._api_url}/projects/{project_id}/repository-logs/split?privacy-aware=false",
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def internal_get_metric_schema(self, project_id, uniformPathType='CODE'):
        """
        """
        response = self.get(
            f"{self._api_url}/projects/{project_id}/metric-schema/{uniformPathType}",
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def internal_get_branch_analysis_state(self, project_id, branchName):
        """
        Note: ignoring branchName parameter
        """
        response = self.get(
            f"{self._api_url}/projects/{project_id}/branch-analysis-state/{branchName}",
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    # ---------- PUBLIC API ----------

    def get_health_check(self, critical_only=True):
        """
        """
        response = self.get(
            f"{self._api_url_version}/health-check?critical-only={str(critical_only)}",
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def get_findings_churn_list(self, project_id, timeStamp, branch, maxCountFindings=-1):
        """
        """
        response = self.get(
            f"{self._api_url_version}/projects/{project_id}/finding-churn/list",
            params={
                "t": f'{branch}:{timeStamp}',
                "max": maxCountFindings
            },
            headers={'Accept': 'application/json',
                     'Content-Type': 'application/json'}
        )
        return response.json()

    def delete_project(self, project_id):
        """
        """
        return self.delete(f"{self._api_url_version}/projects/{project_id}")
