namespace Palmprint.Application.Interfaces;

public interface IOperationLogger
{
    Task LogAsync(Guid tenantId, Guid? userId, string operationType, bool success, double? similarityScore, string? message, string? ipAddress);
}