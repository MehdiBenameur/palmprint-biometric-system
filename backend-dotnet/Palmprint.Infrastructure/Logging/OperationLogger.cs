using Palmprint.Application.Interfaces;
using Palmprint.Domain.Entities;
using Palmprint.Infrastructure.Persistence;

namespace Palmprint.Infrastructure.Logging;

public class OperationLogger : IOperationLogger
{
	private readonly PalmprintDbContext _context;

	public OperationLogger(PalmprintDbContext context)
	{
		_context = context;
	}

	public async Task LogAsync(
		Guid tenantId,
		Guid? userId,
		string operationType,
		bool success,
		double? similarityScore,
		string? message,
		string? ipAddress)
	{
		var log = new BiometricOperationLog
		{
			TenantId = tenantId,
			UserId = userId,
			OperationType = operationType,
			Success = success,
			SimilarityScore = similarityScore,
			Message = message,
			IpAddress = ipAddress
		};

		await _context.BiometricOperationLogs.AddAsync(log);
		await _context.SaveChangesAsync();
	}
}