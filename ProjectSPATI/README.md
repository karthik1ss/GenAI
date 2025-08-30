## ProjectSPATI

** Describe ProjectSPATI here **

This package is a Python Lambda package to use with CDK Pipeline. It doesn't have an API Gateway
definition associated with it. It's most useful when you just want to deploy a lambda function,
perhaps for use as a stream consumer, or invoked by SQS, SNS, or CloudWatch.

In addition to usual Python code and configuration files, this package contains BATS transformation
configurations under `configurations/aws_lambda`. BATS requires these configurations to know how to
transform this package into a deployable Lambda package. If you customize your build system, don't
forget this requirement. Read more about [BATS Lambda Transformer here](https://builderhub.corp.amazon.com/docs/bats/user-guide/transformers-lambda.html).

This package does not contain any deployment logic, they are defined in CDK Package.

## Integrating with existing CDK package

If your CDK package does not have any stages or stacks yet, follow [our guides](https://builderhub.corp.amazon.com/docs/native-aws/developer-guide/cdk-pipeline.html#application-stacks)
to add them to your setup.

Once you have your stack ready, add the sample Lambda function using this snippet:

```
  new lambda.Function(this, 'ProjectSPATIHelloWorldService', {
    code: LambdaAsset.fromBrazil({
      brazilPackage: BrazilPackage.fromString('ProjectSPATI-1.0'),
      componentName: 'HelloWorldService',
    }),
    handler: 'handlers.hello_world',
    memorySize: 128,
    timeout: cdk.Duration.seconds(30),
    runtime: lambda.Runtime.PYTHON_3_8
  });
```

## General Workflow

For testing with this Lambda package, here's our current recommendation:

1. Unit tests. Run good old fashioned unit tests against your code.
1. Deploy to your personal stack and validate the functionalities there. This needs to be done in two steps:
   1. Run `brazil-build` in this package.
   1. Run `brazil-build cdk deploy --hotswap $StackName` in your CDK package.
1. CR and Push. Run integration tests in your pipeline for your function.
