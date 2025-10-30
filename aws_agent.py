import boto3
import datetime

def get_monthly_bill():
    client = boto3.client('ce')  # AWS Cost Explorer
    end = datetime.date.today().replace(day=1)
    start = (end - datetime.timedelta(days=30)).replace(day=1)
    response = client.get_cost_and_usage(
        TimePeriod={'Start': str(start), 'End': str(end)},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost']
    )
    amount = response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']
    return f"Total AWS bill for the month: ${float(amount):.2f}"

def find_idle_ec2_instances():
    ec2 = boto3.client('ec2')
    cw = boto3.client('cloudwatch')
    instances = ec2.describe_instances()
    idle_instances = []

    for reservation in instances['Reservations']:
        for instance in reservation['Instances']:
            id = instance['InstanceId']
            metrics = cw.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': id}],
                StartTime=datetime.datetime.utcnow() - datetime.timedelta(days=7),
                EndTime=datetime.datetime.utcnow(),
                Period=86400,
                Statistics=['Average']
            )
            if not metrics['Datapoints'] or metrics['Datapoints'][0]['Average'] < 5:
                idle_instances.append(id)

    if not idle_instances:
        return "No idle EC2 instances detected this week."
    return f"Idle EC2 instances (CPU <5%): {idle_instances}"
