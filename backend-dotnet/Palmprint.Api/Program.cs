using Microsoft.EntityFrameworkCore;
using Palmprint.Infrastructure.Persistence;
using Palmprint.Application.Interfaces;
using Palmprint.Application.Services;
using Palmprint.Infrastructure.Ai;
using Palmprint.Infrastructure.Logging;
using Palmprint.Infrastructure.Repositories;
using Palmprint.Infrastructure.Security;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

builder.Services.AddDbContext<PalmprintDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddScoped<IEnrollmentService, EnrollmentService>();
builder.Services.AddScoped<IEnrollmentRepository, EnrollmentRepository>();
builder.Services.AddHttpClient<IAiServiceClient, HttpAiServiceClient>();
builder.Services.AddScoped<ITemplateSecurityService, TemplateSecurityService>();
builder.Services.AddScoped<IOperationLogger, OperationLogger>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();