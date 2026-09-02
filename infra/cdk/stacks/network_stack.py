"""Network Stack — VPC, public/private subnets, NAT gateways, IGW, and a
Lambda security group.

Mirrors the reference standard in the parent repo's ``infrastructure.yaml``:
  * VPC 10.0.0.0/16 (DNS hostnames + support enabled)
  * 2 public subnets  (/24) — one per AZ, with an Internet Gateway
  * 2 private subnets (/24) — one per AZ, each with egress via its own NAT GW
  * 2 NAT gateways + 2 Elastic IPs (one per AZ for high availability)
  * A dedicated Lambda security group (all egress allowed)

CDK's high-level ``ec2.Vpc`` construct provisions the VPC, subnets, Internet
Gateway, NAT gateways, Elastic IPs and all route tables / associations
automatically from the subnet configuration below.
"""

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class NetworkStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # VPC with 2 public + 2 private subnets across 2 AZs.
        # nat_gateways=2 => one NAT GW (and one EIP) per AZ, matching the
        # reference standard's two-NAT high-availability layout.
        self.vpc = ec2.Vpc(
            self, "Vpc",
            vpc_name="car-design-explorer-vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            max_azs=2,
            nat_gateways=2,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # Security group for VPC-attached Lambda functions. Egress-only
        # (all outbound) so functions reach AWS service endpoints via NAT.
        self.lambda_security_group = ec2.SecurityGroup(
            self, "LambdaSecurityGroup",
            vpc=self.vpc,
            description="Security group for Car Design Explorer Lambda functions",
            allow_all_outbound=True,
        )
        cdk.Tags.of(self.lambda_security_group).add(
            "Name", "car-design-explorer-lambda-sg"
        )

        # Outputs
        cdk.CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        cdk.CfnOutput(
            self, "PrivateSubnetIds",
            value=",".join(s.subnet_id for s in self.vpc.private_subnets),
        )
        cdk.CfnOutput(
            self, "PublicSubnetIds",
            value=",".join(s.subnet_id for s in self.vpc.public_subnets),
        )
        cdk.CfnOutput(
            self, "LambdaSecurityGroupId",
            value=self.lambda_security_group.security_group_id,
        )
