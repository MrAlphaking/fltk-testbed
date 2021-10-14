import logging
import time
import uuid
from queue import PriorityQueue
from typing import List
import numpy as np

from kubeflow.pytorchjob import PyTorchJobClient
from kubeflow.pytorchjob.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, PYTORCHJOB_PLURAL
from kubernetes import client

from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ArrivalGenerator, Arrival
from fltk.util.task.task import ArrivalTask

AMOUNT_OF_TASKS = 4

# AMOUNT_OF_TASKS needs to match the total amount of jobs im scheduling or it wont start

number_of_jobs = AMOUNT_OF_TASKS
lamda = 0.7
service_rate = 1

jobs = []

# inter_arrival_times = np.random.exponential(1/lamda, number_of_jobs)*10
# job_sizes = np.maximum(np.floor(np.random.exponential(1/service_rate, number_of_jobs) * 3), np.ones(number_of_jobs))
inter_arrival_times = [1,1,1,1]
job_sizes = [1,1,1,1]

arrival_in_q = np.zeros(number_of_jobs)
arrival_in_server = np.zeros(number_of_jobs)
completion_time = np.zeros(number_of_jobs)
current_time = 0

print("number_of_jobs = ", number_of_jobs)

class Orchestrator(object):
    """
    Central component of the Federated Learning System: The Orchestrator

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in the sense that it keeps a central model, performs
    aggregations and keeps track of Clients. For this, the KubeFlow PyTorch-Operator is used to deploy a train task as
    a V1PyTorchJob, which automatically generates the required setup in the cluster. In addition, this allows more Jobs
    to be scheduled, than that there are resources, as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()
    deployed_tasks: List[ArrivalTask] = []
    completed_tasks: List[str] = []

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: BareConfig):
        self.__logger = logging.getLogger('Orchestrator')
        self.__logger.debug("Loading in-cluster configuration")
        self.__cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self.__client = PyTorchJobClient()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self.__logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestartor.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        self._alive = True
        start_time = time.time()
        if clear:
            self.__clear_jobs()

        print("################################# Created index #################################")
        index = 0
        index_arrivals = 0

        print("################################# Starting arrivals loader #################################")
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            # print('Arrive')
            # print(self.__arrival_generator.arrivals.qsize())

            while not self.__arrival_generator.arrivals.empty():
                print(f"################################# In arrivals loop, index = {index_arrivals} #################################")
                arrival: Arrival = self.__arrival_generator.arrivals.get()
                unique_identifier: uuid.UUID = uuid.uuid4()
                #### QUE STUFF #####
                arrival.get_parameter_config().max_epoch = job_sizes[index_arrivals]
                index_arrivals += 1
                #### END QUE STUFF #####
                task = ArrivalTask(priority=arrival.get_priority(),
                                   id=unique_identifier,
                                   network=arrival.get_network(),
                                   dataset=arrival.get_dataset(),
                                   sys_conf=arrival.get_system_config(),
                                   param_conf=arrival.get_parameter_config())
                self.__logger.debug(f"Arrival of: {task}")
                self.pending_tasks.put(task)

            if self.pending_tasks.qsize() != AMOUNT_OF_TASKS:
                # print(self.pending_tasks.qsize())
                # print('CONTINUEEEEEEEEEEEEE')
                continue
            # print('Task')
            value = False
            if self.pending_tasks.qsize() == AMOUNT_OF_TASKS:
                value = True
            # print(PyTorchJobClient.get(namespace='test').__sizeof__())
            # print(self.pending_tasks.qsize())
            # while self.pending_tasks.qsize() != 3:
            #     print(self.pending_tasks.qsize())
            #     temp = self.pending_tasks.get()

            #### QUE STUFF #####
            print("################################# Starting que stuff #################################")
            server_start_time = time.time() # Start timer on deployment start
            time.sleep(inter_arrival_times[index]) # Wait the first interarrival time
            arrival_in_q[0] = time.time() - server_start_time   # log the first entry to the q
            #### END QUE STUFF #####

            while not self.pending_tasks.empty():
                #### QUE STUFF #####
                arrival_in_server[index] = time.time() - server_start_time  # log the first entry to the server
                #### END QUE STUFF #####
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()
                self.__logger.info(f"Scheduling arrival of Arrival: {curr_task.id}")
                job_to_start = construct_job(self._config, curr_task)

                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                if value:
                    self.deployed_tasks.append(curr_task)
                    self.__logger.info(f"Deploying on cluster: {curr_task.id}")
                    self.__client.create(job_to_start, namespace=self._config.cluster_config.namespace)

                # TODO: Extend this logic in your real project, this is only meant for demo purposes
                # For now we exit the thread after scheduling a single task.
                # self.stop()
                # return

                #### QUE STUFF #####
                # If we finished all the jobs stop, to avoid index out of bounds etc
                if index == number_of_jobs:
                    print("Job: i, arrival in q time: time, arrival in server time: time, completion time: time")
                    for i in range(number_of_jobs):
                        print(f"{i} {arrival_in_q[i]} {arrival_in_server[i]} {completion_time[i]}")
                    self.stop()
                # Wait until job completes
                print("################################# (Not) Sleeping 100 #################################")
                # time.sleep(100) # This is here because checking job status too early may cause it to crash
                print(f"self.__client.get(namespace=self._config.cluster_config.namespace)['items'][0]['metadata']['name']: {self.__client.get(namespace=self._config.cluster_config.namespace)['items'][0]['metadata']['name']}")
                job_name = self.__client.get(namespace=self._config.cluster_config.namespace)['items'][0]['metadata']['name'] #+ "-master-0"
                print(f"self.__client.get_job_status = {self.__client.get_job_status(job_name, namespace=self._config.cluster_config.namespace)}")
                print(
                    f"self.__client.is_job_running(name, namespace) = {self.__client.is_job_running(job_name, namespace=self._config.cluster_config.namespace)}")

                while self.__client.get_job_status(job_name, namespace=self._config.cluster_config.namespace) != "Succeeded":
                    time.sleep(1)
                completion_time[index] = time.time() - server_start_time # Log when the job was completed
                print("################################# I survived scary loop! #################################") # to test that this is not an inf loop
                index += 1 # inc our job-index
                print(f"index = {index}")
                # If the next job was supposed to enter the system already
                if time.time() - arrival_in_q[index-1] >= inter_arrival_times[index]:
                    arrival_in_q[index] = arrival_in_q[index-1] + inter_arrival_times[index]
                # If the next job was has still "not arrived" according to timer, wait the reminder
                if time.time() - arrival_in_q[index-1] < inter_arrival_times[index]:
                    time.sleep(inter_arrival_times[index] - (time.time() - arrival_in_q[index-1]))
                #### END QUE STUFF #####

            self.__logger.debug("Still alive...")
            # print(self.deployed_tasks.__sizeof__())
            if value:
                self.stop()
            time.sleep(5)

        logging.info(f'Experiment completed, currently does not support waiting.')

    def __clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests)
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self.__logger.info(f'Clearing old jobs in current namespace: {namespace}')

        for job in self.__client.get(namespace=self._config.cluster_config.namespace)['items']:
            job_name = job['metadata']['name']
            self.__logger.info(f'Deleting: {job_name}')
            try:
                self.__client.custom_api.delete_namespaced_custom_object(
                    PYTORCHJOB_GROUP,
                    PYTORCHJOB_VERSION,
                    namespace,
                    PYTORCHJOB_PLURAL,
                    job_name)
            except Exception as e:
                self.__logger.warning(f'Could not delete: {job_name}')
                print(e)